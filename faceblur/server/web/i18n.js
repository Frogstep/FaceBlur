// Web app localization. Strings in app.js stay in English and double as the
// lookup key: t("Hello {name}", { name }). Plurals go through tn(), which knows
// Russian's three forms. Wording follows the desktop app (faceblur/gui/i18n.py).

export const LANGUAGES = { en: "English", ru: "Русский" };

const KEY = "faceblur.lang";
let lang = initial();

function initial() {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved && LANGUAGES[saved]) return saved;
  } catch { /* storage blocked */ }
  // the browser's most preferred language that we have
  for (const l of navigator.languages || [navigator.language || "en"]) {
    const code = l.slice(0, 2).toLowerCase();
    if (LANGUAGES[code]) return code;
  }
  return "en";
}

export function language() {
  return lang;
}

export function setLanguage(code) {
  lang = LANGUAGES[code] ? code : "en";
  try { localStorage.setItem(KEY, lang); } catch { /* storage blocked */ }
  document.documentElement.lang = lang;
}

document.documentElement.lang = lang;

function fill(text, vars) {
  return vars ? text.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : text;
}

export function t(text, vars) {
  const table = lang === "ru" ? RU : null;
  return fill((table && table[text]) || text, vars);
}

// tn(n, "{n} frame", "{n} frames"): `one` is also the key of the Russian forms
export function tn(n, one, many, vars = {}) {
  const forms = lang === "ru" ? RU_PLURAL[one] : null;
  let text;
  if (!forms) text = n === 1 ? one : many;
  else {
    const n10 = Math.abs(n) % 10, n100 = Math.abs(n) % 100;
    text = n10 === 1 && n100 !== 11 ? forms[0]
      : n10 >= 2 && n10 <= 4 && !(n100 >= 12 && n100 <= 14) ? forms[1] : forms[2];
  }
  return fill(text, { n, ...vars });
}

const RU = {
  // header, sign in
  "Sign out": "Выйти",
  "Sign in": "Войти",
  "Sign in to FaceBlur": "Вход в FaceBlur",
  "Access token": "Токен доступа",
  "Paste the token that faceblur serve printed (it is also in server_token.txt), or open the sign-in link it printed.":
    "Вставьте токен, который вывела команда faceblur serve (он также записан в server_token.txt), или откройте ссылку для входа, которую она вывела.",
  "That token is not right.": "Неверный токен.",
  "Signed out.": "Вы вышли.",
  "The sign-in link is not valid (the token changed?).": "Ссылка для входа недействительна (токен изменился?).",
  "The server is not reachable.": "Сервер недоступен.",
  "Please sign in.": "Войдите, пожалуйста.",
  "Try again": "Повторить",
  "Language": "Язык",

  // statuses and stages
  "Uploading": "Загрузка",
  "Waiting to detect faces": "Ожидает поиска лиц",
  "Detecting faces": "Поиск лиц",
  "Label the faces": "Разметьте лица",
  "Waiting to render": "Ожидает рендера",
  "Rendering": "Рендер",
  "Safety check": "Проверка безопасности",
  "Done": "Готово",
  "Failed": "Ошибка",
  "Cancelled": "Отменено",
  "Loading face models": "Загрузка моделей лиц",
  "Preparing face cards": "Подготовка карточек лиц",
  "Loading SAM 2": "Загрузка SAM 2",
  // messages the server puts on a job
  "Face detection was cancelled.": "Поиск лиц отменён.",
  "Rendering was cancelled.": "Рендер отменён.",
  "The server stopped during the face scan. Start it again.": "Сервер остановился во время поиска лиц. Запустите его снова.",
  "The server stopped during the render. Start it again.": "Сервер остановился во время рендера. Запустите его снова.",

  // labels
  "No label": "Без метки",
  "Someone else": "Кто-то другой",
  "Not a face": "Не лицо",
  "Person {k}": "Человек {k}",

  // uploads
  "{name} is {size}; the server accepts up to {max}.": "{name}: {size}, а сервер принимает до {max}.",
  "{name} is already uploading.": "{name} уже загружается.",
  "Resuming {name} at {size}.": "Продолжаем загрузку {name} с {size}.",
  "The connection keeps failing. Choose the same file again to resume.":
    "Соединение всё время обрывается. Выберите тот же файл ещё раз, чтобы продолжить.",
  "Connection problem, retrying ({k}/8)…": "Проблема со связью, повтор ({k}/8)…",
  "{name} uploaded. Detecting faces…": "{name} загружен. Ищем лица…",
  "Cancel": "Отмена",

  // jobs list
  "Drop a video here": "Перетащите видео сюда",
  "or choose one (up to {max}). Large files upload in parts and can be resumed.":
    "или выберите файл (до {max}). Большие файлы загружаются частями, загрузку можно продолжить.",
  "Choose video": "Выбрать видео",
  "Videos": "Видео",
  "Loading…": "Загрузка…",
  "No videos yet.": "Видео пока нет.",
  "{done} of {size} — choose the file again to resume": "{done} из {size} — выберите файл ещё раз, чтобы продолжить",
  "#{k} in queue": "#{k} в очереди",
  "Video": "Видео",
  "Status": "Статус",
  "Length": "Длина",
  "Added": "Добавлено",
  "Delete": "Удалить",
  "Delete {name} and its result?": "Удалить {name} вместе с результатом?",

  // job page
  "This video was deleted.": "Это видео удалено.",
  "Back to the list": "К списку",
  "← Videos": "← Видео",
  "no audio": "без звука",
  "Detect again": "Найти лица заново",
  "Discard the detections and labels and scan the video again":
    "Сбросить найденные лица и метки и просканировать видео заново",
  "Detect faces again? All face labels are cleared.": "Найти лица заново? Все метки лиц будут сброшены.",
  "The upload stopped at {done} of {size}. Choose the same file to continue.":
    "Загрузка остановилась на {done} из {size}. Выберите тот же файл, чтобы продолжить.",
  "Choose the file": "Выбрать файл",
  "Waiting: #{k} in the queue": "Ожидание: #{k} в очереди",

  // result
  "This browser cannot play the video (it is H.265). Download it and play it in another app.":
    "Этот браузер не может воспроизвести видео (H.265). Скачайте его и откройте в другом приложении.",
  "The safety check was skipped.": "Проверка безопасности пропущена.",
  "Safety check: no masked face is recognizable in the result.":
    "Проверка безопасности: ни одно скрытое лицо в результате не узнаваемо.",
  "Previous result": "Предыдущий результат",
  "Result": "Результат",
  "Download": "Скачать",
  "Frame {f} ({time}), similarity {sim}": "Кадр {f} ({time}), сходство {sim}",
  "{time} · frame {f}": "{time} · кадр {f}",
  "…and {n} more.": "…и ещё {n}.",

  // review
  "Faces": "Лица",
  "Each card is one face followed continuously through the video. Say who every card is: a person to mask, someone else (stays visible) or not a face. One person usually has several cards. Similar faces are listed next to each other.":
    "Каждая карточка — одно лицо, которое непрерывно отслеживается в видео. Укажите, кто на каждой карточке: человек, которого нужно скрыть, кто-то другой (остаётся видимым) или не лицо. У одного человека обычно несколько карточек. Похожие лица стоят рядом.",
  "Keys:": "Клавиши:",
  "person": "человек",
  "someone else": "кто-то другой",
  "not a face": "не лицо",
  "clear": "очистить",
  "select all": "выбрать все",
  "view frame": "показать кадр",
  "All cards": "Все карточки",
  "Not labeled": "Не размечено",
  "Rest:": "Остальные:",
  "Label every card without a label as someone else": "Пометить все неразмеченные карточки как «Кто-то другой»",
  "Label every card without a label as not a face": "Пометить все неразмеченные карточки как «Не лицо»",
  "People to mask:": "Кого скрыть:",
  "Click to rename": "Нажмите, чтобы переименовать",
  "Name": "Имя",
  "Remove (only when no card is labeled with this person)": "Удалить (только если ни одна карточка не помечена этим человеком)",
  "+ Person": "+ Человек",
  "Key {k}": "Клавиша {k}",
  "New person": "Новый человек",
  "Clear": "Очистить",
  "Every card has a label.": "У всех карточек есть метка.",
  "No faces were found in this video.": "В этом видео лица не найдены.",
  "Select card {n}": "Выбрать карточку {n}",
  "View on the video (Enter)": "Показать на видео (Enter)",
  "View": "Смотреть",
  "Select cards, then:": "Выберите карточки, затем:",
  "All {n} cards labeled": "Все карточки размечены ({n})",
  "No faces to mask.": "Нечего скрывать: лица не найдены.",
  "Label every card first ({n} left).": "Сначала разметьте все карточки (осталось {n}).",
  "Label at least one card as a person to mask.": "Пометьте хотя бы одну карточку как человека, которого нужно скрыть.",
  "Render": "Рендер",
  "Render again": "Рендер заново",
  "Masks every person's cards, then runs the safety check.":
    "Скрывает лица отмеченных людей, затем запускает проверку безопасности.",
  "Every card already has a label.": "У всех карточек уже есть метка.",
  "Label the {n} card(s) without a label as “{label}”?": "Пометить неразмеченные карточки ({n}) как «{label}»?",

  // settings
  "Mask": "Маска",
  "Style": "Стиль",
  "Pixelate": "Пикселизация",
  "Blur": "Размытие",
  "Solid fill": "Заливка",
  "Strength": "Сила",
  "Margin around the face": "Отступ вокруг лица",
  "Exact face outline (SAM 2)": "Точный контур лица (SAM 2)",
  "Off: ellipses, much faster": "Выкл.: эллипсы, намного быстрее",

  // frame viewer
  "Card #{n}": "Карточка #{n}",
  "Frame {f} · {time}": "Кадр {f} · {time}",
  "Close": "Закрыть",
  "◀ Card": "◀ Карточка",
  "Card ▶": "Карточка ▶",
  "Previous sample of this card": "Предыдущий образец этой карточки",
  "Next sample of this card": "Следующий образец этой карточки",
  "#{n} is:": "#{n} — это:",
  "Boxes are numbered by card and coloured by label (yellow: no label yet).":
    "Рамки пронумерованы по карточкам и окрашены по меткам (жёлтая — ещё без метки).",
  "video frame": "кадр видео",
};

const RU_PLURAL = {
  "{n} frame": ["{n} кадр", "{n} кадра", "{n} кадров"],
  "{n} selected:": ["Выбрана {n}:", "Выбрано {n}:", "Выбрано {n}:"],
  "{n} of {total} cards without a label": [
    "Без метки {n} карточка из {total}", "Без метки {n} карточки из {total}", "Без метки {n} карточек из {total}"],
  "A masked face may still be recognizable on {n} frame. Check it below; fix hard frames in the desktop app if needed.": [
    "Скрытое лицо может быть узнаваемо на {n} кадре. Проверьте его ниже; сложные кадры при необходимости исправьте в настольном приложении.",
    "Скрытое лицо может быть узнаваемо на {n} кадрах. Проверьте их ниже; сложные кадры при необходимости исправьте в настольном приложении.",
    "Скрытое лицо может быть узнаваемо на {n} кадрах. Проверьте их ниже; сложные кадры при необходимости исправьте в настольном приложении."],
};
