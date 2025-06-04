import logging
import tensorflow as tf
import re
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, ConversationHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
import pickle
import os
import numpy as np
import csv
from datetime import datetime

MODEL_PATH = "news_model_3 copy.keras"
model = tf.keras.models.load_model(MODEL_PATH)

TOKEN = ""
ALLOWED_USERS = {
    675684022: "admin",
}

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

(
    CHOOSING_MODE,
    TESTING_MODE,
    TRAINING_MODE,
    STANDARD_MODE,
    CHOOSING_ROLE,
    AWAITING_TEXT,
    AWAITING_FEEDBACK,
    AWAITING_RATING,
    AWAITING_CORRECTION,
    AWAITING_LONGER_TEXT
) = range(10)

with open('C:/Диплом/tokenizer_3.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

tokenizer = Tokenizer(num_words=30000, oov_token="<UNK>")
tokenizer.__dict__.update({
    'config': tokenizer_data['config'],
    'word_index': tokenizer_data['word_index'],
    'index_word': tokenizer_data['index_word']
})


def log_feedback(user_id: int, text: str, model_score: float, user_feedback: str, role: str):
    with open('feedback_log.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_id,
            role,
            text[:200],
            f"{model_score:.1f}%",
            user_feedback
        ])


def is_text_too_short(text: str, min_words: int = 20) -> bool:
    words = wordpunct_tokenize(text)
    return len(words) < min_words


def preprocess_text(text: str, max_len: int = 32) -> tf.Tensor:

    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    tokens = wordpunct_tokenize(text.lower())
    processed_tokens = [
        morph.parse(word)[0].normal_form
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    sequence = tokenizer.texts_to_sequences([' '.join(processed_tokens)])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return tf.convert_to_tensor(padded, dtype=tf.int32)


async def start(update: Update, context: CallbackContext):
    reply_markup = ReplyKeyboardMarkup(
        [["🔍 Стандартная проверка", "🧪 Тестовый режим", "🎓 Режим дообучения"]],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await update.message.reply_text(
        "🔍 Выберите режим работы:\n\n"
        "• <b>Стандартная проверка</b> - обычная проверка новости\n"
        "• <b>Тестовый режим</b> - оценка качества работы модели\n"
        "• <b>Режим дообучения</b> - улучшение модели (только для админов)",
        parse_mode='HTML',
        reply_markup=reply_markup
    )
    return CHOOSING_MODE


async def choose_mode(update: Update, context: CallbackContext):
    mode = update.message.text
    user_id = update.message.from_user.id

    if mode == "🎓 Режим дообучения":
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text(
                "⛔ Только администраторы могут использовать этот режим",
                reply_markup=ReplyKeyboardRemove()
            )
            return await start(update, context)
        context.user_data['mode'] = 'training'
        await update.message.reply_text(
            "Режим дообучения активирован. Присылайте новости с правильными оценками.",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_TEXT

    elif mode == "🧪 Тестовый режим":
        context.user_data['mode'] = 'testing'
        await update.message.reply_text(
            "Тестовый режим. Пожалуйста, оценивайте ответы по шкале от 1 до 5.",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_TEXT

    else:
        context.user_data['mode'] = 'standard'
        await update.message.reply_text(
            "Стандартный режим проверки. Присылайте новость для анализа.",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_TEXT


async def handle_longer_text(update: Update, context: CallbackContext):
    text = update.message.text

    if is_text_too_short(text):
        await update.message.reply_text(
            "⚠️ Текст все еще слишком короткий. Пожалуйста, пришлите новость длиннее (минимум 10 слов)."
        )
        return AWAITING_LONGER_TEXT

    return await handle_text(update, context)


async def handle_text(update: Update, context: CallbackContext):
    text = update.message.text
    mode = context.user_data.get('mode', 'standard')

    if is_text_too_short(text):
        await update.message.reply_text(
            "⚠️ Текст слишком короткий для анализа. Пожалуйста, пришлите более подробную новость (минимум 10 слов)."
        )
        return AWAITING_LONGER_TEXT

    try:
        processed = preprocess_text(text)
        prediction = model.predict(processed)
        confidence = float(prediction[0][0]) * 100

        context.user_data['last_text'] = text
        context.user_data['last_prediction'] = confidence
        context.user_data['last_processed'] = processed.numpy()

        verdict = ("✅ Высокая достоверность" if confidence > 70 else
                   "⚠️ Возможна дезинформация" if confidence > 30 else
                   "❌ Низкая достоверность")

        if mode == 'testing':
            reply_markup = ReplyKeyboardMarkup(
                [["1", "2", "3", "4", "5"], ["🚫 Пропустить"]],
                resize_keyboard=True
            )
            message = (f"{verdict}\nОценка модели: {confidence:.1f}%\n"
                       "Оцените точность (1-5):")
            await update.message.reply_text(message, reply_markup=reply_markup)
            return AWAITING_RATING

        elif mode == 'training':
            reply_markup = ReplyKeyboardMarkup(
                [["✅ Верно", "✏️ Исправить"], ["🚫 Пропустить"]],
                resize_keyboard=True
            )
            message = (f"{verdict}\nТекущая оценка: {confidence:.1f}%\n"
                       "Подтвердить или указать верную оценку:")
            await update.message.reply_text(message, reply_markup=reply_markup)
            return AWAITING_FEEDBACK

        else:
            message = f"{verdict}\nОценка достоверности: {confidence:.1f}%"
            await update.message.reply_text(message)
            return AWAITING_TEXT

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await update.message.reply_text("⚠️ Ошибка обработки. Попробуйте другой текст.")
        return AWAITING_TEXT


async def handle_rating(update: Update, context: CallbackContext):
    feedback = update.message.text

    if feedback == "🚫 Пропустить":
        await update.message.reply_text("Пропущено", reply_markup=ReplyKeyboardRemove())
        return AWAITING_TEXT

    if feedback in ["1", "2", "3", "4", "5"]:
        log_feedback(
            user_id=update.message.from_user.id,
            text=context.user_data['last_text'],
            model_score=context.user_data['last_prediction'],
            user_feedback=f"RATING_{feedback}",
            role="user"
        )
        await update.message.reply_text(
            "Спасибо за оценку! Это поможет улучшить бота.",
            reply_markup=ReplyKeyboardRemove()
        )
    else:
        await update.message.reply_text(
            "Пожалуйста, выберите оценку от 1 до 5",
            reply_markup=ReplyKeyboardMarkup(
                [["1", "2", "3", "4", "5"], ["🚫 Пропустить"]],
                resize_keyboard=True
            )
        )
        return AWAITING_RATING

    return AWAITING_TEXT


async def handle_feedback(update: Update, context: CallbackContext):
    feedback = update.message.text

    if feedback == "🚫 Пропустить":
        await update.message.reply_text("Пропущено", reply_markup=ReplyKeyboardRemove())
        return AWAITING_TEXT

    if feedback == "✅ Верно":
        log_feedback(
            user_id=update.message.from_user.id,
            text=context.user_data['last_text'],
            model_score=context.user_data['last_prediction'],
            user_feedback="VERIFIED",
            role="admin"
        )
        await update.message.reply_text(
            "Спасибо за подтверждение!",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_TEXT

    elif feedback == "✏️ Исправить":
        await update.message.reply_text(
            "Укажите правильную оценку достоверности (0-100):",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_CORRECTION
    else:
        await update.message.reply_text(
            "Пожалуйста, используйте кнопки для ответа",
            reply_markup=ReplyKeyboardMarkup(
                [["✅ Верно", "✏️ Исправить"], ["🚫 Пропустить"]],
                resize_keyboard=True
            )
        )
        return AWAITING_FEEDBACK


async def handle_correction(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    user_input = update.message.text.strip()

    try:
        correction = float(user_input.replace(',', '.'))
        if not 0 <= correction <= 100:
            raise ValueError("Число вне диапазона")

        processed = context.user_data['last_processed']

        if processed.ndim == 3:
            processed = processed.reshape(processed.shape[0], -1)
        elif processed.ndim == 1:
            processed = processed.reshape(1, -1)

        model.fit(
            processed,
            np.array([[correction/100.0]]),
            epochs=1,
            verbose=0
        )
        model.save(MODEL_PATH)

        log_feedback(
            user_id=user_id,
            text=context.user_data['last_text'],
            model_score=context.user_data['last_prediction'],
            user_feedback=f"CORRECTED_TO_{correction}",
            role="admin"
        )

        await update.message.reply_text(
            f"✅ Модель обновлена! Новая оценка: {correction:.1f}%",
            reply_markup=ReplyKeyboardRemove()
        )
        return AWAITING_TEXT

    except ValueError as e:
        logging.error(f"Ошибка: {str(e)}")
        await update.message.reply_text(
            "⚠️ Введите число 0-100 (пример: 50, 75.5 или 100)\n"
            "Ошибка: " + str(e)
        )
        return AWAITING_CORRECTION


async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text(
        "Диалог завершен. Нажмите /start для новой проверки.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


def main() -> None:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING_MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_mode)],
            AWAITING_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)],
            AWAITING_RATING: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_rating)],
            AWAITING_FEEDBACK: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_feedback)],
            AWAITING_CORRECTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_correction)],
            AWAITING_LONGER_TEXT: [MessageHandler(
                filters.TEXT & ~filters.COMMAND, handle_longer_text)]
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True
    )

    application.add_handler(conv_handler)

    if not os.path.exists('feedback_log.csv'):
        with open('feedback_log.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'User ID', 'Role',
                            'Text', 'Model Score', 'User Feedback'])

    application.run_polling()


if __name__ == "__main__":
    main()
