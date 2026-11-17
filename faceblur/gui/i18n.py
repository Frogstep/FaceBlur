"""GUI localization.

Strings in the code stay in English and double as the lookup key:
``tr("Found {n} face cards", n=3)``. Plural phrases go through ``trn``, which
knows that Russian has three plural forms. The language is chosen once per
window (see MainWindow._set_language); it is stored in QSettings.
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "ru": "Русский"}

_lang = "en"


def set_language(code: str) -> None:
    global _lang
    _lang = code if code in LANGUAGES else "en"


def language() -> str:
    return _lang


def system_language() -> str:
    from PySide6.QtCore import QLocale

    return "ru" if QLocale.system().language() == QLocale.Language.Russian else "en"


def tr(text: str, *args, **kw) -> str:
    """Translate `text`, then fill in {placeholders} (unused ones are ignored)."""
    out = _TABLES.get(_lang, {}).get(text, text)
    return out.format(*args, **kw) if (args or kw) else out


def trn(n: int, one: str, many: str, **kw) -> str:
    """Plural form for n: `one`/`many` are English templates that may use {n}."""
    forms = _PLURALS.get(_lang, {}).get(one)
    if forms is None:
        text = one if n == 1 else many
    else:  # Russian: 1 / 2-4 / 5-20 (with the usual 11-14 exception)
        n10, n100 = abs(n) % 10, abs(n) % 100
        text = forms[0] if n10 == 1 and n100 != 11 else (
            forms[1] if 2 <= n10 <= 4 and not 12 <= n100 <= 14 else forms[2])
    return text.format(n=n, **kw)


HELP_EN = (
    "1. File > Open a video. Faces are detected once and cached.\n\n"
    "2. In the Review faces tab, label every face card: a person to mask, someone else "
    "(not masked) or not a face. A card is one face followed continuously; one person usually "
    "has several cards. Select cards and press 1-9, O or N.\n\n"
    "3. Scrub the timeline. Coloured bands show where each person will be masked "
    "(bright = face detected, dim = bridged gap or drawn box).\n\n"
    "4. Fix mistakes on the video:\n"
    "   • click a box → label its card, or 'Not this person' for one frame or track\n"
    "   • drag a rectangle → 'Mask this area' for a person (SAM 2 tracks it from there)\n\n"
    "5. Render. A safety check then looks for any masked face that is still recognizable; "
    "flagged frames are listed and marked red on the timeline.\n\n"
    "Keys: Space play/pause, ←/→ frame, Shift+←/→ one second, Home/End.\n"
    "The project (people, settings, corrections) is saved next to the video as .faceblur.json.")

_RU = {
    # ---- transport / video side
    "Cancel": "Отмена",
    "Previous second (Shift+Left)": "Назад на секунду (Shift+←)",
    "Previous frame (Left)": "Предыдущий кадр (←)",
    "Play / pause (Space)": "Воспроизведение / пауза (Пробел)",
    "Next frame (Right)": "Следующий кадр (→)",
    "Next second (Shift+Right)": "Вперёд на секунду (Shift+→)",
    "Original": "Оригинал",
    "Rendered output": "Результат",
    "Preview mask": "Предпросмотр маски",
    "Show the mask on the original (box-based; the render also uses SAM 2)":
        "Показать маску на оригинале (по рамкам; при рендере также используется SAM 2)",
    "Show face boxes": "Рамки лиц",
    "Click a face box to choose who it is or to exclude it. Drag a rectangle to mask an area the detector missed.":
        "Щёлкните по рамке лица, чтобы указать, кто это, или исключить её. "
        "Протяните прямоугольник, чтобы скрыть область, которую пропустил детектор.",
    "Video": "Видео",
    "Review faces": "Проверка лиц",
    "Open a video to start (File > Open, or drop a file here)":
        "Откройте видео, чтобы начать (Файл > Открыть или перетащите файл сюда)",
    "   frame {f} / {last}": "   кадр {f} / {last}",

    # ---- side panel
    "People to mask": "Кого скрывать",
    "Add person": "Добавить человека",
    "Face review": "Проверка лиц",
    "Open a video to detect faces.": "Откройте видео, чтобы найти лица.",
    "Open the face review": "Открыть проверку лиц",
    "Mask": "Маска",
    "Pixelate": "Пикселизация",
    "Blur (pixelated first)": "Размытие (поверх пикселизации)",
    "Solid black": "Чёрная заливка",
    "Higher = coarser blocks": "Больше = крупнее блоки",
    "Extra margin around each face box": "Дополнительный отступ вокруг каждой рамки лица",
    "Precise outline with SAM 2 (slower)": "Точный контур с SAM 2 (медленнее)",
    "Style": "Стиль",
    "Strength": "Сила",
    "Margin": "Отступ",
    "Corrections": "Исправления",
    "Drawn box covers ±": "Нарисованная рамка действует ±",
    " frames": " кадр.",
    "Remove selected correction": "Удалить выбранное исправление",
    "Processing": "Обработка",
    "Device": "Устройство",
    "Auto (GPU if available)": "Авто (GPU, если есть)",
    "GPU (NVIDIA)": "GPU (NVIDIA)",
    "CPU only (slow)": "Только CPU (медленно)",
    "Where face detection, SAM 2 and encoding run. CPU works on any PC but is much slower.":
        "Где выполняются поиск лиц, SAM 2 и кодирование. CPU работает на любом ПК, но гораздо медленнее.",
    "No usable NVIDIA GPU found; everything runs on the CPU (slow).":
        "Подходящая видеокарта NVIDIA не найдена; всё выполняется на CPU (медленно).",
    "No usable NVIDIA GPU found. Choose Auto or CPU.":
        "Подходящая видеокарта NVIDIA не найдена. Выберите «Авто» или «CPU».",
    "faces": "лица",
    "decode": "декодирование",
    "encode": "кодирование",
    "Render": "Рендер",
    "Output file": "Выходной файл",
    "Choose the output file": "Выбрать выходной файл",
    "Render  (Ctrl+R)": "Рендер  (Ctrl+R)",
    "Label {n} more {cards} to render": "Разметьте ещё карточек: {n}",
    "Safety check": "Проверка безопасности",
    "Runs automatically after rendering.": "Запускается автоматически после рендера.",
    "Run safety check after rendering": "Проверка безопасности после рендера",
    "Re-detect faces in the output and flag frames where a masked face may still be recognizable (slower)":
        "Повторно найти лица в результате и отметить кадры, где скрытое лицо всё ещё узнаваемо (медленнее)",
    "Off. Tick “Run safety check after rendering” to run it.":
        "Выключена. Отметьте «Проверка безопасности после рендера», чтобы запустить её.",
    "Skipped (turned off for this render).": "Пропущена (выключена для этого рендера).",

    # ---- menus
    "&File": "&Файл",
    "&Open video or project…": "&Открыть видео или проект…",
    "&Save project": "&Сохранить проект",
    "Save project &as…": "Сохранить проект &как…",
    "Detect faces again (discard cache)": "Найти лица заново (сбросить кэш)",
    "E&xit": "В&ыход",
    "&Language": "&Язык",
    "&Help": "&Справка",
    "How to use": "Как пользоваться",
    "Language": "Язык",
    "Finish or cancel the current job before changing the language.":
        "Дождитесь окончания текущей задачи или отмените её, прежде чем менять язык.",

    # ---- files
    "Videos": "Видео",
    "All files": "Все файлы",
    "Open video or project": "Открыть видео или проект",
    "Video or project": "Видео или проект",
    "Cannot open": "Не удалось открыть",
    "Save failed": "Не удалось сохранить",
    "Saved {path}": "Сохранено: {path}",
    "Save project": "Сохранить проект",
    "FaceBlur project": "Проект FaceBlur",
    "Unsaved changes": "Несохранённые изменения",
    "Save changes to the project first?": "Сначала сохранить изменения в проекте?",
    "Busy": "Идёт работа",
    "A job is running. Cancel it and quit?": "Выполняется задача. Отменить её и выйти?",
    "Output video": "Выходное видео",
    "MP4 video": "Видео MP4",

    # ---- detection
    "Detecting faces": "Поиск лиц",
    "Detecting faces…": "Поиск лиц…",
    "Loading face models": "Загрузка моделей лиц",
    "Preparing face cards": "Подготовка карточек лиц",
    "Loading SAM 2": "Загрузка SAM 2",
    "Rendering": "Рендер",
    "Note: variable frame rate video; the output will use a constant frame rate.":
        "Примечание: у видео переменная частота кадров; результат будет с постоянной частотой.",
    "Found {cards} in {frames} frames.": "Найдено карточек лиц: {n} (кадров в видео: {frames}).",
    "Detect faces again": "Найти лица заново",
    "Discard cached detections and scan the whole video again?\n"
    "Face cards will change, so all face labels and corrections are cleared.":
        "Удалить кэш распознавания и заново просканировать всё видео?\n"
        "Карточки лиц изменятся, поэтому все метки и исправления будут сброшены.",
    "Detecting faces: {pct}% ({done} of {total} frames)": "Поиск лиц: {pct}% (кадров: {done} из {total})",
    ", about {left} left.": ", осталось около {left}.",
    "\nYou can scrub the video meanwhile; results are cached for next time.":
        "\nПока можно просматривать видео; результаты сохранятся в кэш.",
    "Face detection was cancelled. Use File › Detect faces again to restart it.":
        "Поиск лиц отменён. Чтобы запустить его снова, выберите Файл › Найти лица заново.",
    "starting…": "запуск…",
    "%p%  ·  frame {done} / {total}": "%p%  ·  кадр {done} / {total}",
    "  ·  {rate} fps  ·  about {left} left": "  ·  {rate} к/с  ·  осталось около {left}",
    "{s} s": "{s} с",
    "{m} min {s} s": "{m} мин {s} с",
    "{h} h {m} min": "{h} ч {m} мин",
    "{label} cancelled.": "{label}: отменено.",
    "Failed: ": "Ошибка: ",
    "Error": "Ошибка",
    "Cancelling…": "Отмена…",

    # ---- people
    "Remove this person": "Удалить этого человека",
    "Nobody is masked yet. Label face cards in the\nReview faces tab, or click a face in the video.":
        "Пока никто не скрыт. Разметьте карточки лиц на\nвкладке «Проверка лиц» или щёлкните по лицу в видео.",
    "Person {k}": "Человек {k}",
    "Remove person": "Удалить человека",
    "Remove {name}? Their {n_cards} face card(s) become unlabeled again and {n_manual} drawn box(es) are removed.":
        "Удалить «{name}»? Карточки лиц этого человека ({n_cards}) снова станут неразмеченными, "
        "а нарисованные рамки ({n_manual}) будут удалены.",
    "No faces found. You can still drag boxes onto the video to mask areas.":
        "Лица не найдены. Вы всё равно можете рисовать рамки на видео, чтобы скрыть области.",
    "<b>{done} of {total}</b> face cards labeled. ":
        "Размечено карточек лиц: <b>{done} из {total}</b>. ",
    "Label the remaining {n} to render.": "Разметьте оставшиеся ({n}), чтобы начать рендер.",
    "All <b>{total}</b> face cards labeled.": "Все карточки лиц размечены (<b>{total}</b>).",
    "Review faces ({n} to label)": "Проверка лиц (не размечено: {n})",
    "Review faces ✓": "Проверка лиц ✓",

    # ---- corrections / overlays / context menus
    "whole track": "весь трек",
    "this frame": "только этот кадр",
    "✕  {time} (frame {frame}): not masked, {what}": "✕  {time} (кадр {frame}): не скрывать, {what}",
    "▢  {time} (frame {frame}): {name}, ±{span} frames": "▢  {time} (кадр {frame}): {name}, ±{span} кадр.",
    "still recognizable {s}": "всё ещё узнаваемо {s}",
    "✕ not masked {tag}": "✕ не скрыто {tag}",
    "(nearby)": "(рядом)",
    "{tag} ? not labeled": "{tag} ? не размечено",
    "someone else": "кто-то другой",
    "not a face": "не лицо",
    "{name} (bridged)": "{name} (заполнено)",
    "{name} (drawn)": "{name} (нарисовано)",
    "a new person": "нового человека",
    "Mask as {}": "Скрыть как: {}",
    "for {}": "для: {}",
    "Drawn box for {name}, frame {frame} ±{span}": "Нарисованная рамка: {name}, кадр {frame} ±{span}",
    "Remove this drawn box": "Удалить эту рамку",
    "Mask this face again (undo “not this person”)": "Снова скрывать это лицо (отменить «не этот человек»)",
    "Face card #{num} (the whole card)": "Карточка лица #{num} (вся карточка)",
    "Someone else (not masked)": "Кто-то другой (не скрывать)",
    "Not a face": "Не лицо",
    "Clear the label": "Снять метку",
    "Show this card in the review": "Показать карточку в проверке лиц",
    "Not this person — this frame only": "Не этот человек — только этот кадр",
    "Not this person — whole track": "Не этот человек — весь трек",
    "Every frame where this face is followed continuously": "Все кадры, где это лицо непрерывно отслеживается",
    "Mask this face here too": "Скрыть и это лицо здесь",
    "Mask this area (±{n} frames)": "Скрыть эту область (±{n} кадр.)",

    # ---- render / safety
    "Faces to review": "Лица на проверку",
    "{n} face card(s) have no label yet. Every card needs one: a person (masked), someone else, or not a face.":
        "Карточек лиц без метки: {n}. Каждой карточке нужна метка: человек (скрыть), кто-то другой или не лицо.",
    "Nothing to mask": "Нечего скрывать",
    "No card is labeled as a person to mask, and there are no drawn boxes.":
        "Ни одна карточка не отмечена как человек, которого нужно скрыть, и нарисованных рамок нет.",
    "Output": "Выходной файл",
    "Choose an output file first.": "Сначала выберите выходной файл.",
    "The output must be a different file from the original video.":
        "Выходной файл должен отличаться от исходного видео.",
    "Overwrite": "Перезапись",
    "{name} exists. Overwrite it?": "Файл {name} уже существует. Перезаписать?",
    "Waiting for render…": "Ожидание рендера…",
    "Wrote {name}: {frames} frames at {fps} fps. Frames with no mask at all: {unmasked}.":
        "Записан файл {name} (кадров: {frames}, {fps} к/с). Кадров совсем без маски: {unmasked}.",
    "Output found; no safety check on record for it.": "Результат найден, но проверки безопасности для него нет.",
    "Passed.": "Пройдено.",
    "No masked face is recognizable on any output frame.":
        "Ни одно скрытое лицо не узнаётся ни на одном кадре результата.",
    "{n} frame(s) need a look.": "Кадров, требующих внимания: {n}.",
    "A masked face may still be recognizable. Click to view, fix with a drawn box, and render again.":
        "Скрытое лицо может всё ещё узнаваться. Щёлкните, чтобы посмотреть, исправьте нарисованной рамкой "
        "и запустите рендер снова.",
    "{time} (frame {frame}) — similarity {sim}": "{time} (кадр {frame}) — сходство {sim}",

    # ---- review tab
    "Not labeled": "Не размечено",
    "{name} (masked)": "{name} (скрыть)",
    "Double-click to show this frame in the video": "Двойной щелчок — показать этот кадр в видео",
    "Click a card to select it (Ctrl+click: several, Shift+click: a range), then label it with a button below "
    "or a key: <b>1–9</b> person, <b>O</b> someone else, <b>N</b> not a face, <b>Backspace</b> clear. "
    "<b>Space</b> jumps to the next unlabeled card. Double-click a thumbnail to see it in the video. "
    "Similar faces are listed next to each other; nothing is joined automatically.":
        "Щёлкните по карточке, чтобы выбрать её (Ctrl+щелчок — несколько, Shift+щелчок — диапазон), затем "
        "присвойте метку кнопкой ниже или клавишей: <b>1–9</b> человек, <b>O</b> кто-то другой, <b>N</b> не лицо, "
        "<b>Backspace</b> снять метку. <b>Пробел</b> — к следующей неразмеченной карточке. Двойной щелчок по "
        "миниатюре показывает её в видео. Похожие лица стоят рядом; автоматически ничего не объединяется.",
    "Show": "Показать",
    "All cards": "Все карточки",
    "Masked people": "Скрываемые люди",
    "No faces were found in this video.": "В этом видео лица не найдены.",
    "{done} of {total} face cards labeled": "Размечено карточек лиц: {done} из {total}",
    "{n} still need a label before rendering.": "Ещё {n} нужно разметить перед рендером.",
    "All {total} face cards labeled.": "Все карточки лиц размечены ({total}).",
    "Ready to render.": "Можно запускать рендер.",
    "New person": "Новый человек",
    "Mask the selected cards as a new person": "Скрыть выбранные карточки как нового человека",
    "Someone else (O)": "Кто-то другой (O)",
    "A real face that is not masked": "Настоящее лицо, которое не нужно скрывать",
    "Not a face (N)": "Не лицо (N)",
    "A false detection": "Ложное срабатывание",
    "Clear": "Очистить",
    "Remove the label": "Снять метку",
    "frames: {n}": "кадров: {n}",

    # ---- help
    HELP_EN: (
        "1. Файл > Открыть видео. Лица ищутся один раз и сохраняются в кэш.\n\n"
        "2. На вкладке «Проверка лиц» присвойте метку каждой карточке лица: человек, которого нужно скрыть, "
        "кто-то другой (не скрывается) или не лицо. Карточка — это одно лицо, которое непрерывно отслеживается; "
        "у одного человека обычно несколько карточек. Выберите карточки и нажмите 1-9, O или N.\n\n"
        "3. Просмотрите временную шкалу. Цветные полосы показывают, где каждый человек будет скрыт "
        "(ярко — лицо найдено, тускло — заполненный пропуск или нарисованная рамка).\n\n"
        "4. Исправляйте ошибки прямо на видео:\n"
        "   • щелчок по рамке → метка её карточки или «Не этот человек» для кадра или трека\n"
        "   • протяните прямоугольник → «Скрыть эту область» для человека (дальше его отслеживает SAM 2)\n\n"
        "5. Рендер. Затем проверка безопасности ищет скрытые лица, которые всё ещё можно узнать; "
        "такие кадры выводятся списком и отмечаются красным на шкале.\n\n"
        "Клавиши: Пробел — воспроизведение/пауза, ←/→ — кадр, Shift+←/→ — секунда, Home/End.\n"
        "Проект (люди, настройки, исправления) сохраняется рядом с видео как .faceblur.json."),
}

_TABLES = {"ru": _RU}

# English singular template -> Russian forms for 1 / 2-4 / 5+
_PLURALS = {
    "ru": {
        "{n} card": ("{n} карточка", "{n} карточки", "{n} карточек"),
        ", {n} drawn box": (", {n} рамка", ", {n} рамки", ", {n} рамок"),
        "{n} face card": ("{n} карточка лица", "{n} карточки лица", "{n} карточек лица"),
    },
}
