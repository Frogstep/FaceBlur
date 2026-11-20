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
    "1. File > Open a video. Before faces are detected you can cut and crop it: drag the corners of the "
    "frame to crop, split the timeline at the playhead (S) and remove the parts you don't need (Delete), "
    "or cut the start and end (I / O). Then click Detect faces: faces are only looked for in what is "
    "left, and they are cached.\n\n"
    "2. In the Review faces tab, every face starts as not blurred. Click the faces you want to blur; "
    "click again to leave a face visible. A card is one face followed continuously, so one person "
    "can have several cards: click each of them.\n\n"
    "3. Scrub the timeline. The red band shows where faces will be blurred "
    "(bright = face detected, dim = bridged gap or drawn area).\n\n"
    "4. Fix mistakes on the video:\n"
    "   • click a box → blur that face or not, or 'Don't blur here' for one frame or track\n"
    "   • drag a rectangle → 'Blur this area' (SAM 2 tracks it from there)\n\n"
    "5. Render. A safety check then looks for any blurred face that is still recognizable; "
    "flagged frames are listed and marked red on the timeline.\n\n"
    "Keys: Space play/pause, ←/→ frame, Shift+←/→ one second, Home/End.\n"
    "The project (faces to blur, settings, corrections) is saved next to the video as .faceblur.json.")

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
    "Click a face box to blur it or leave it visible. Drag a rectangle to blur an area the detector missed.":
        "Щёлкните по рамке лица, чтобы размыть его или оставить видимым. "
        "Протяните прямоугольник, чтобы размыть область, которую пропустил детектор.",
    "Video": "Видео",
    "Review faces": "Проверка лиц",
    "Open a video to start (File > Open, or drop a file here)":
        "Откройте видео, чтобы начать (Файл > Открыть или перетащите файл сюда)",
    "   frame {f} / {last}": "   кадр {f} / {last}",

    # ---- side panel
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
    "Open video or project…": "Открыть видео или проект…",
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
    "Saving the video": "Сохранение видео",
    "Note: variable frame rate video; the output will use a constant frame rate.":
        "Примечание: у видео переменная частота кадров; результат будет с постоянной частотой.",
    "Found {cards} in {frames} frames.": "Найдено карточек лиц: {n} (кадров в видео: {frames}).",
    "Detect faces again": "Найти лица заново",
    "Discard cached detections and scan the whole video again?\n"
    "Face cards will change, so all face labels and corrections are cleared.":
        "Удалить кэш распознавания и заново просканировать всё видео?\n"
        "Карточки лиц изменятся, поэтому выбор лиц для размытия и исправления будут сброшены.",
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

    # ---- blur choice
    "No faces found. You can still drag boxes onto the video to blur areas.":
        "Лица не найдены. Вы всё равно можете рисовать рамки на видео, чтобы размыть области.",
    "Click the faces to blur in the Review faces tab.": "Нажмите на лица, которые нужно размыть, на вкладке «Проверка лиц».",
    "Review faces ({n} blurred)": "Проверка лиц (размыть: {n})",

    # ---- corrections / overlays / context menus
    "whole track": "весь трек",
    "this frame": "только этот кадр",
    "✕  {time} (frame {frame}): not masked, {what}": "✕  {time} (кадр {frame}): не размывать, {what}",
    "▢  {time} (frame {frame}): blurred area, ±{span} frames": "▢  {time} (кадр {frame}): размытая область, ±{span} кадр.",
    "still recognizable {s}": "всё ещё узнаваемо {s}",
    "{tag} not blurred here": "{tag} здесь не размывается",
    "{tag} blur": "{tag} размыть",
    "blur (nearby)": "размыть (рядом)",
    "blur (bridged)": "размыть (заполнено)",
    "blur (drawn)": "размыть (нарисовано)",
    "Drawn area, frame {frame} ±{span}": "Нарисованная область, кадр {frame} ±{span}",
    "Remove this drawn area": "Удалить эту область",
    "Blur this face here again": "Снова размывать это лицо здесь",
    "Blur face #{num}": "Размыть лицо #{num}",
    "Don't blur face #{num}": "Не размывать лицо #{num}",
    "Show face #{num} in the list": "Показать лицо #{num} в списке",
    "Don't blur here — this frame only": "Не размывать здесь — только этот кадр",
    "Don't blur here — while followed continuously": "Не размывать здесь — пока лицо непрерывно отслеживается",
    "Every frame where this face is followed continuously": "Все кадры, где это лицо непрерывно отслеживается",
    "Blur this face here too (±{n} frames)": "Размыть и здесь (±{n} кадр.)",
    "Blur this area (±{n} frames)": "Размыть эту область (±{n} кадр.)",

    # ---- render / safety
    "Nothing to blur": "Нечего размывать",
    "No face is marked to blur yet. Click the faces to blur in the Review faces tab, or drag a box onto the video.":
        "Ни одно лицо пока не выбрано для размытия. Нажмите на нужные лица на вкладке «Проверка лиц» "
        "или нарисуйте рамку на видео.",
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
    "Click to blur this face, or to leave it visible": "Щёлкните, чтобы размыть это лицо или оставить его видимым",
    "Show in video": "Показать в видео",
    "Show this face in the video": "Показать это лицо в видео",
    "BLURRED": "РАЗМЫТЬ",
    "Not blurred": "Не размывать",
    "Every face stays visible until you choose to blur it. <b>Click a face to blur it</b>; click it again to "
    "leave it visible. Keys: <b>↑/↓</b> move, <b>Space</b> switch, <b>Enter</b> show in the video.":
        "Все лица остаются видимыми, пока вы не решите их размыть. <b>Щёлкните по лицу, чтобы размыть его</b>; "
        "щёлкните ещё раз, чтобы оставить его видимым. Клавиши: <b>↑/↓</b> — переход, <b>Пробел</b> — "
        "переключить, <b>Enter</b> — показать в видео.",
    "Blur all": "Размыть все",
    "Blur none": "Убрать размытие",
    "Remove the blur from all {n} faces?": "Убрать размытие со всех лиц ({n})?",
    "Show": "Показать",
    "All faces": "Все лица",
    "Blurred": "Размыть",
    "No faces were found in this video.": "В этом видео лица не найдены.",

    # ---- cut & crop
    "⇤ Start here": "⇤ Начать здесь",
    "Cut everything before the playhead (I)": "Вырезать всё до позиции воспроизведения (I)",
    "End here ⇥": "Закончить здесь ⇥",
    "Cut everything after the playhead (O)": "Вырезать всё после позиции воспроизведения (O)",
    "Split": "Разрезать",
    "Undo": "Отменить",
    "Redo": "Повторить",
    "Split the part under the playhead (S)": "Разрезать часть в позиции воспроизведения (S)",
    "Remove part": "Удалить часть",
    "Bring part back": "Вернуть часть",
    "Remove the part under the playhead, or bring it back (Delete)":
        "Удалить часть в позиции воспроизведения или вернуть её (Delete)",
    "Crop": "Кадр",
    "Free": "Свободно",
    "Shape of the crop": "Пропорции кадра",
    "Full frame": "Весь кадр",
    "Remove the crop": "Убрать кадрирование",
    "Undo (Ctrl+Z)": "Отменить (Ctrl+Z)",
    "Redo (Ctrl+Y)": "Повторить (Ctrl+Y)",
    "Crop: drag the corners or edges of the frame (drag outside it to draw a new crop). "
    "Cut: split at the playhead (S) and remove the parts you don't need (Delete), or cut the "
    "start and end (I / O). Faces are only looked for in what is left.":
        "Кадр: тяните за углы или края кадра (протяните снаружи, чтобы нарисовать новый). "
        "Обрезка: разрежьте в позиции воспроизведения (S) и удалите ненужные части (Delete) или обрежьте "
        "начало и конец (I / O). Лица ищутся только в том, что осталось.",
    "Cut & crop": "Обрезка и кадр",
    "Change cut & crop…": "Изменить обрезку и кадр…",
    "Detect faces  ▶": "Найти лица  ▶",
    "Back to the faces (nothing changed)": "Вернуться к лицам (без изменений)",
    "Cut and crop the video if you need to, then click Detect faces.":
        "При необходимости обрежьте и кадрируйте видео, затем нажмите «Найти лица».",
    "Changing the cut or crop means detecting the faces again. This clears your choices "
    "({faces} chosen to blur, {fixes} corrections). Continue?":
        "После изменения обрезки или кадра лица придётся искать заново. Ваш выбор будет сброшен "
        "(выбрано для размытия: {faces}, исправлений: {fixes}). Продолжить?",
    "At least one part of the video must stay.": "Хотя бы одна часть видео должна остаться.",
    "Keeps {kept} of {total} ({parts})": "Остаётся {kept} из {total} ({parts})",
    "The whole video ({total})": "Всё видео ({total})",
    "cropped to {w} × {h}": "кадр {w} × {h}",
    "full frame {w} × {h}": "весь кадр {w} × {h}",
    "  · cut": "  · вырезано",

    # ---- reset
    "Reset project…": "Сбросить проект…",
    "Reset project": "Сброс проекта",
    "Start over with this video: deletes the detected faces, your choices, corrections, cut & crop and the "
    "work folder": "Начать с этим видео заново: удаляет найденные лица, ваш выбор, исправления, обрезку и кадр, "
                   "а также рабочую папку",
    "Reset the project for {name}?\n\n"
    "This deletes the detected faces, your choice of faces to blur, the corrections, "
    "the cut & crop and the settings, and the work folder:\n{work}\n\n"
    "The video itself and rendered videos are kept. This cannot be undone.":
        "Сбросить проект для {name}?\n\n"
        "Будут удалены найденные лица, выбор лиц для размытия, исправления, обрезка и кадр, настройки, "
        "а также рабочая папка:\n{work}\n\n"
        "Само видео и готовые результаты останутся. Отменить это нельзя.",
    "Could not delete everything:": "Не удалось удалить всё:",
    "Project reset: {name} is open as a new video.": "Проект сброшен: {name} открыто как новое видео.",

    # ---- help
    HELP_EN: (
        "1. Файл > Открыть видео. До поиска лиц видео можно обрезать: потяните за углы кадра, чтобы "
        "кадрировать, разрежьте шкалу в позиции воспроизведения (S) и удалите ненужные части (Delete) "
        "или обрежьте начало и конец (I / O). Затем нажмите «Найти лица»: лица ищутся только в "
        "оставшейся части и сохраняются в кэш.\n\n"
        "2. На вкладке «Проверка лиц» все лица сначала не размыты. Нажмите на лица, которые нужно размыть; "
        "повторное нажатие оставляет лицо видимым. Карточка — это одно лицо, которое непрерывно отслеживается, "
        "поэтому у одного человека может быть несколько карточек: нажмите на каждую.\n\n"
        "3. Просмотрите временную шкалу. Красная полоса показывает, где лица будут размыты "
        "(ярко — лицо найдено, тускло — заполненный пропуск или нарисованная область).\n\n"
        "4. Исправляйте ошибки прямо на видео:\n"
        "   • щелчок по рамке → размыть это лицо или нет, либо «Не размывать здесь» для кадра или трека\n"
        "   • протяните прямоугольник → «Размыть эту область» (дальше её отслеживает SAM 2)\n\n"
        "5. Рендер. Затем проверка безопасности ищет размытые лица, которые всё ещё можно узнать; "
        "такие кадры выводятся списком и отмечаются красным на шкале.\n\n"
        "Клавиши: Пробел — воспроизведение/пауза, ←/→ — кадр, Shift+←/→ — секунда, Home/End.\n"
        "Проект (выбор лиц, настройки, исправления) сохраняется рядом с видео как .faceblur.json."),
}

_TABLES = {"ru": _RU}

# English singular template -> Russian forms for 1 / 2-4 / 5+
_PLURALS = {
    "ru": {
        "{n} face card": ("{n} карточка лица", "{n} карточки лица", "{n} карточек лица"),
        "{n} frame": ("{n} кадр", "{n} кадра", "{n} кадров"),
        "{n} part": ("{n} часть", "{n} части", "{n} частей"),
        "{n} of {total} faces will be blurred": ("Размыть {n} лицо из {total}", "Размыть {n} лица из {total}",
                                                "Размыть {n} лиц из {total}"),
        "<b>{n}</b> of {total} faces will be blurred.": ("Размыть <b>{n}</b> лицо из {total}.",
                                                       "Размыть <b>{n}</b> лица из {total}.",
                                                       "Размыть <b>{n}</b> лиц из {total}."),
        "Plus {n} drawn area.": ("Плюс {n} нарисованная область.", "Плюс {n} нарисованные области.",
                                 "Плюс {n} нарисованных областей."),
    },
}
