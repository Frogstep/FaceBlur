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
  "Choose faces to blur": "Выберите лица для размытия",
  "Waiting to render": "Ожидает рендера",
  "Rendering": "Рендер",
  "Saving the video": "Сохранение видео",
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
  "Discard the detected faces and the choice of faces to blur, and scan the video again":
    "Сбросить найденные лица и выбор лиц для размытия и просканировать видео заново",
  "Detect faces again? The choice of faces to blur is cleared.": "Найти лица заново? Выбор лиц для размытия будет сброшен.",
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
  "Every face stays visible until you choose to blur it. Click a face to blur it; click it again to leave it visible.":
    "Все лица остаются видимыми, пока вы не решите их размыть. Нажмите на лицо, чтобы размыть его; нажмите ещё раз, чтобы оставить его видимым.",
  "Show": "Показать",
  "All faces": "Все лица",
  "Blurred": "Размыть",
  "Not blurred": "Не размывать",
  "Blur all": "Размыть все",
  "Blur none": "Убрать размытие",
  "Remove the blur from all {n} faces?": "Убрать размытие со всех лиц ({n})?",
  "No faces were found in this video.": "В этом видео лица не найдены.",
  "No face is blurred yet.": "Пока нет лиц с отметкой «Размыть».",
  "Every face is blurred.": "Все лица отмечены «Размыть».",
  "Face #{n}: blurred": "Лицо #{n}: размыть",
  "Face #{n}: not blurred": "Лицо #{n}: не размывать",
  "Show this face on the video": "Показать это лицо на видео",
  "View": "Смотреть",
  "No faces to blur.": "Нет лиц для размытия.",
  "Click the faces you want to blur.": "Нажмите на лица, которые нужно размыть.",
  "Render": "Рендер",
  "Render again": "Рендер заново",

  // settings
  "Settings": "Настройки",
  "Style": "Стиль",
  "Pixelate": "Пикселизация",
  "Blur": "Размытие",
  "Solid fill": "Заливка",
  "Strength": "Сила",
  "Margin around the face": "Отступ вокруг лица",
  "Exact face outline (SAM 2)": "Точный контур лица (SAM 2)",
  "Slower. Off: ellipses, much faster": "Медленнее. Выкл.: эллипсы, намного быстрее",
  "Run safety check after rendering": "Проверка безопасности после рендера",
  "Looks for blurred faces that are still recognizable in the result (slower)":
    "Ищет в результате размытые лица, которые всё ещё можно узнать (медленнее)",
  "The safety check is off (turn it on in the settings below).": "Проверка безопасности выключена (её можно включить в настройках ниже).",

  // frame viewer
  "Face #{n}": "Лицо #{n}",
  "Frame {f} · {time}": "Кадр {f} · {time}",
  "Close": "Закрыть",
  "Blur #{n}": "Размыть #{n}",
  "Don't blur #{n}": "Не размывать #{n}",
  "◀ Earlier": "◀ Раньше",
  "Later ▶": "Позже ▶",
  "An earlier picture of this face": "Более ранний кадр с этим лицом",
  "A later picture of this face": "Более поздний кадр с этим лицом",
  "Red boxes are blurred, green boxes stay visible. The numbers are the face numbers in the list.":
    "Красные рамки — размытые лица, зелёные — остаются видимыми. Номера совпадают с номерами лиц в списке.",
  "video frame": "кадр видео",
};

const RU_PLURAL = {
  "{n} frame": ["{n} кадр", "{n} кадра", "{n} кадров"],
  "{n} of {total} faces blurred": ["Размыть {n} лицо из {total}", "Размыть {n} лица из {total}", "Размыть {n} лиц из {total}"],
  "Blurs {n} face.": ["Размоет {n} лицо.", "Размоет {n} лица.", "Размоет {n} лиц."],
  "Blurs {n} face, then runs the safety check.": [
    "Размоет {n} лицо, затем запустит проверку безопасности.",
    "Размоет {n} лица, затем запустит проверку безопасности.",
    "Размоет {n} лиц, затем запустит проверку безопасности."],
  "A masked face may still be recognizable on {n} frame. Check it below; fix hard frames in the desktop app if needed.": [
    "Скрытое лицо может быть узнаваемо на {n} кадре. Проверьте его ниже; сложные кадры при необходимости исправьте в настольном приложении.",
    "Скрытое лицо может быть узнаваемо на {n} кадрах. Проверьте их ниже; сложные кадры при необходимости исправьте в настольном приложении.",
    "Скрытое лицо может быть узнаваемо на {n} кадрах. Проверьте их ниже; сложные кадры при необходимости исправьте в настольном приложении."],
};
