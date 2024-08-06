from telebot.asyncio_handler_backends import StatesGroup, State


class MyStates(StatesGroup):
    process_description = State()
    process_mask = State()
    transcription = State()
    ai_chat = State()
    ai_chat_select_style = State()
    ai_chat_selected_style = State()
    ai_images = State()


async def split_text(text, max_length):
    parts = []
    start = 0
    max_length_for_only = max_length

    while start < len(text):
        end = start + max_length

        # Проверяем, чтобы конец не выходил за пределы текста
        if len(text) <= end:
            end = len(text)
        else:
            if max_length == max_length_for_only:
                max_length = max_length - 4
            end = start + max_length

        parts.append(text[start:end])

        start = end

    return parts


async def format_messages(parts):
    formatted_messages = []
    number_parts = len(parts)
    if number_parts == 1:
        return parts

    for i, part in enumerate(parts, 1):
        formatted_messages.append(f"{i}/{number_parts}\n{part}")

    return formatted_messages
