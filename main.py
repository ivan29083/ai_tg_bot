# https://www.youtube.com/watch?v=lvOCm88mQiQ    деплой на сервер, консоль
import logging
from os import path

import asyncio

from telebot.async_telebot import AsyncTeleBot, Handler
from telebot.asyncio_storage import StateMemoryStorage

from telebot import asyncio_filters
from telebot import types

from openai import AsyncOpenAI

import requests
import speech_recognition as sr
import soundfile as sf
from io import BytesIO
from PIL import Image

from add_file import split_text, format_messages, MyStates

try:
    from settings import test_bot_token, ai_bot_token, openai_key, чат_со_мной_id, helpfull_prompt, classic_prompt
except Exception as ex:
    print(f'Возникла проблема с импортом констант настроек из модуля "settings.py". Проверьте его заполненность или создайте модуль '
          f'со своими значениями констант.\nОшибка - {ex}')


BOT_TOKEN = ai_bot_token

state_storage = StateMemoryStorage()  # you can init here another storage

bot = AsyncTeleBot(BOT_TOKEN, state_storage=state_storage)

welcome_message = 'Привет, я твой собеседник с искусственным интеллектом. Могу отвечать тебе на текстовые и аудио сообщения. Пиши в любой момент)'

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=openai_key,
)


################################################
def logging_exceptions(func):
    # декоратор для логирования async функции
    async def wrapper(*args, **kwargs):
        logger.info(f'{func.__name__}:  старт функции')
        try:
            res = await func(*args, **kwargs)
            logger.info(f'{func.__name__}: успешно выполнена')
        except Exception as exception:
            logger.error(f'{func.__name__}: при выполнении ошибка {exception}')
            raise
        return res

    return wrapper


def logging_exceptions_sync(func):
    # декоратор для логирования sync функции
    def wrapper(*args, **kwargs):
        logger.info(f'{func.__name__}:  старт функции')
        try:
            res = func(*args, **kwargs)
            logger.info(f'{func.__name__}: успешно выполнена')
        except Exception as exception:
            logger.error(f'{func.__name__}: при выполнении ошибка {exception}')
            raise
        return res

    return wrapper
###################################################


@logging_exceptions
async def generate_response(text):

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        model="gpt-4o-mini",
        # stream=True,
    )

    if response and response.choices:
        res = response.choices[0].message.content.strip()

        print(f"Функция - (generate_response)\nAnswer: {res}")
        return res
    else:
        return


@logging_exceptions
@bot.message_handler(commands=['start'])
async def start(message):
    await bot.send_message(message.chat.id,
                           f'Привет, @{message.from_user.username}!\n{welcome_message}')


@logging_exceptions
@bot.message_handler(commands=['help'])
async def help(message):
    await bot.send_message(message.chat.id, welcome_message)


@logging_exceptions
@bot.message_handler(commands=['ai_chat'])
async def start_ai_chat(message):
    await bot.set_state(message.from_user.id, MyStates.ai_chat_select_style, message.chat.id)
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton('Классический помощник', callback_data='ai_chat_classic'))
    markup.add(types.InlineKeyboardButton('Понимающе-поддерживающий собседник', callback_data='ai_chat_support'))
    # markup.add(types.InlineKeyboardButton('Другое. Опишу промпт самостоятельно', callback_data='ai_chat_unic'))
    await bot.send_message(чат_со_мной_id, 'Бот используется в режиме общения с Chat-GPT. Выберите стиль последующего разговора,'
                                           'что предварительно настроит вашего собеседника на нужный лад', reply_markup=markup)


@logging_exceptions
@bot.callback_query_handler(func=lambda call: True, state=MyStates.ai_chat_select_style)
async def callback(call):

    chat_id = call.message.chat.id
    if call.data == 'ai_chat_classic':
        await bot.set_state(call.message.chat.id, MyStates.ai_chat_selected_style, call.message.chat.id)
        await start_ai_chat_2(chat_id, 'ai_chat_classic')

    elif call.data == 'ai_chat_support':
        await bot.set_state(call.message.chat.id, MyStates.ai_chat_selected_style, call.message.chat.id)
        await start_ai_chat_2(chat_id, 'ai_chat_support')

    elif call.data == 'ai_chat_unic':
        await bot.send_message(чат_со_мной_id, 'Напиши свой промпт')
        await bot.set_state(call.message.chat.id, MyStates.ai_chat_selected_style, call.message.chat.id)


@logging_exceptions
@bot.message_handler(state=MyStates.ai_chat_selected_style, content_types=['text'])
async def start_ai_chat_2(chat_id, style):
    await bot.set_state(chat_id, MyStates.ai_chat, chat_id)

    if style == 'ai_chat_support':
        content = helpfull_prompt
    else:
        content = classic_prompt

    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": content}
        ],
        model="gpt-4o-mini",
    )

    if response and response.choices:
        text = response.choices[0].message.content.strip()
    else:
        text = 'Ошибка подключения к gpt. Обратитесь к администратору'

    await bot.send_message(chat_id, text)


@logging_exceptions
@bot.message_handler(commands=['transcription'])
async def start_transcription(message):
    await bot.set_state(message.from_user.id, MyStates.transcription, message.chat.id)

    await bot.send_message(message.chat.id, 'Бот в режиме расшифровки аудио. Для смены режима просто выберите другую команду в меню.')


@logging_exceptions
@bot.message_handler(commands=['ai_images'])
async def start_ai_images(message):
    await bot.set_state(message.from_user.id, MyStates.ai_images, message.chat.id)

    await bot.send_message(message.chat.id, 'Бот в режиме работы с изображениями. Для смены режима просто выберите другую команду в меню.'
                                            '\nА сейчас просто введите описание изображения, которое хотите сгенерировать.')


@logging_exceptions
@bot.message_handler(state=MyStates.ai_chat, content_types=['text', 'voice'])
async def input_words_ai_chat(message):
    if message.voice:
        text = await get_text_from_audio(message)
    else:
        text = message.text.strip()

    response = await generate_response(text)
    if response is None:
        await bot.send_message(message.chat.id, 'Ответ не получен, обратитесь к администратору')
        return

    parts = await split_text(response, 4000)
    formatted_messages = await format_messages(parts)
    for formatted_message in formatted_messages:
        await bot.send_message(message.chat.id, formatted_message)
    print(f"Функция - (input_words_ai_chat)\nAnswer for {message.chat.username}: {response}")


@logging_exceptions
async def get_text_from_audio(message):
    file_info = await bot.get_file(message.voice.file_id)
    if message.from_user.username:
        unic = message.from_user.username
    else:
        unic = message.chat.id
    file_name = f'downloads\\{unic}_voice'
    audio_downloaded = download_audio(file_name, file_info)

    if not audio_downloaded:
        text_error = 'Не удалось скачать файл, попробуйте позднее или обратитесь к администратору'
        print(text_error)
        await bot.send_message(message.chat.id, text_error)

    audio_converted = convert_audio(file_name)
    if not audio_converted:
        text_error = 'Не удалось конвертировать файл, попробуйте позднее или обратитесь к администратору'
        print(text_error)
        await bot.send_message(message.chat.id, text_error)

    text = transcrib_audio(file_name)

    return text


@logging_exceptions_sync
def download_audio(file_name, file_info):
    try:
        file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(BOT_TOKEN, file_info.file_path))

        with open(f'{file_name}.ogg', 'wb') as f:
            f.write(file.content)
        return True
    except Exception as e:
        print(e)
        return False


@logging_exceptions_sync
def convert_audio(file_name):
    try:
        data, samplerate = sf.read(f'{file_name}.ogg')
        sf.write(f'{file_name}.wav', data, samplerate)
        return True
    except Exception as e:
        print(e)
        return False


@logging_exceptions_sync
def transcrib_audio(file_name):
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), f"{file_name}.wav")

    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)

    # Perform speech recognition using Google Web Speech API
    try:
        text = r.recognize_google(audio, language="ru-RU")
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return False
    except sr.RequestError as e:
        print(f"Error: Could not request results from Google Speech Recognition service  --- {e}")
        return False


@logging_exceptions
@bot.message_handler(func=lambda message: True, state=MyStates.transcription, content_types=['audio', 'voice'])
async def input_audio(message):
    text = await get_text_from_audio(message)

    if text:
        await bot.send_message(message.chat.id, text)
        print("You said:", text)


@logging_exceptions
@bot.message_handler(state=MyStates.ai_images, content_types=['text', 'voice'])
async def input_words_ai_images(message):
    if message.voice:
        prompt = await get_text_from_audio(message)
    else:
        prompt = message.text.strip()

    response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

    try:
        image_url = response.data[0].url
        result = image_url
    except Exception as ex:
        result = f'При генерации изображения произошла ошибка - {ex}. Обратитесь к администратору.'

    await bot.send_photo(message.chat.id, result)


@logging_exceptions
@bot.message_handler(content_types=['photo', 'document'], state=MyStates.ai_images)
async def handle_photo(message):
    # Запрашиваем описание от пользователя
    await bot.reply_to(message, "Пожалуйста, напишите описание изменений, которые вы хотите увидеть на фотографии.")

    # Ожидаем следующее сообщение от пользователя с описанием
    await bot.set_state(message.from_user.id, MyStates.process_description, message.chat.id)

    async with bot.retrieve_data(message.from_user.id, message.chat.id) as data:
        data['message_with_photo'] = message


@logging_exceptions
@bot.message_handler(state=MyStates.process_description)
async def process_description(message):
    description = message.text
    user_id = message.from_user.id

    # Получаем фотографию от пользователя
    async with bot.retrieve_data(message.from_user.id, message.chat.id) as data:
        message_with_photo = data['message_with_photo']
        # message_with_mask = data['message_with_mask']

    await bot.set_state(message.from_user.id, MyStates.ai_images, message.chat.id)

    byte_array_image = await process_image(message, message_with_photo, 'image')
    # byte_array_mask = await process_image(message, message_with_mask, 'mask')

    response = await client.images.edit(
        model="dall-e-2",
        image=byte_array_image,
        # image=open('imageRomanov_Ivan_image.png', "rb"),
        # mask=open("mask.png", "rb"),
        # mask=byte_array_mask,
        prompt=description,
        n=1,
        size="512x512"
    )
    image_url = response.data[0].url
    await bot.send_photo(user_id, image_url)


@logging_exceptions
async def process_image(message, message_with_photo, suf):
    if message_with_photo.document:
        photo_id = message_with_photo.document.file_id
    else:
        photo_id = message_with_photo.photo[-1].file_id

    file_info = await bot.get_file(photo_id)
    if message.from_user.username:
        unic = message.from_user.username
    else:
        unic = message.chat.id
    file_name = f'downloads\\{suf}{unic}_image'
    image_downloaded = download_image(file_name, file_info)

    if not image_downloaded:
        text_error = 'Не удалось скачать изображение, попробуйте позднее или обратитесь к администратору'
        print(text_error)
        await bot.send_message(message.chat.id, text_error)

    # Read the image file from disk and resize it
    image = Image.open(f'{file_name}.png').convert('RGBA')
    width, height = 512, 512
    image = image.resize((width, height))

    # Convert the image to a BytesIO object
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')
    byte_array = byte_stream.getvalue()

    return byte_array


@logging_exceptions_sync
def download_image(file_name, file_info):
    try:
        photo_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"

        # Загружаем фотографию
        photo_response = requests.get(photo_url)
        with open(f'{file_name}.png', 'wb') as f:
            f.write(photo_response.content)
        return True
    except Exception as e:
        print(e)
        return False


@logging_exceptions
@bot.message_handler()
async def input_words_all(message):

    response = await generate_response(message.text.strip())
    if response is None:
        await bot.send_message(message.chat.id, 'Ответ не получен, обратитесь к администратору')
        return

    parts = await split_text(response, 4000)
    formatted_messages = await format_messages(parts)

    for formatted_message in formatted_messages:
        await bot.send_message(message.chat.id, formatted_message)

    print(f"Функция - (input_words_all)\nAnswer for {message.chat.username}: {response}")


bot.add_custom_filter(asyncio_filters.StateFilter(bot))


async def main():
    await asyncio.gather(bot.infinity_polling())


def start_logging():
    global logger

    logger = logging.getLogger(__name__)
    # создаём хендлер для файла лога, кодировка файла будет UTF-8 для поддержки кириллических сообщений в логе
    filehandler = logging.FileHandler(filename='application_log.log', encoding='utf-8')
    # задаём базовую конфигурацию логирования
    logging.basicConfig(format='[%(levelname)-10s] %(asctime)-25s - %(message)s', handlers=[filehandler],
                        level=logging.INFO)
    logger.warning(f'************************** Бот запущен *****************************')


if __name__ == '__main__':
    # получаем логгер
    start_logging()
    asyncio.run(main())
