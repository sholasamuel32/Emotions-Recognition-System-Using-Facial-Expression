(() => {
  const form = document.getElementById("analysisForm");
  if (!form) return;

  const textArea = document.getElementById("moodText");
  const textInputPanel = document.getElementById("textInputPanel");
  const faceInputPanel = document.getElementById("faceInputPanel");
  const characterCount = document.getElementById("characterCount");
  const modeButtons = document.querySelectorAll("[data-mode]");
  const microphoneButton = document.getElementById("microphoneButton");
  const analyseButton = document.getElementById("analyseButton");
  const errorBox = document.getElementById("analysisError");
  const csrfToken = document.querySelector('meta[name="csrf-token"]').content;

  const cameraVideo = document.getElementById("cameraVideo");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const cameraPlaceholder = document.getElementById("cameraPlaceholder");
  const facePreview = document.getElementById("facePreview");
  const startCameraButton = document.getElementById("startCameraButton");
  const captureButton = document.getElementById("captureButton");
  const faceFile = document.getElementById("faceFile");

  let inputMethod = "text";
  let recognition = null;
  let listening = false;
  let cameraStream = null;
  let faceImageData = "";
  let faceSource = "upload";

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = false;
  };

  const updateCount = () => {
    characterCount.textContent = `${textArea.value.length.toLocaleString()} / 2,000`;
  };
  textArea.addEventListener("input", updateCount);

  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      textArea.value = button.dataset.example;
      updateCount();
      textArea.focus();
    });
  });

  const stopCamera = () => {
    if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    cameraVideo.srcObject = null;
    cameraVideo.hidden = true;
    captureButton.hidden = true;
    startCameraButton.textContent = "● Start camera";
    if (!faceImageData) cameraPlaceholder.hidden = false;
  };

  const updateMode = (button) => {
    inputMethod = button.dataset.mode;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
    const faceMode = inputMethod === "face";
    textInputPanel.hidden = faceMode;
    faceInputPanel.hidden = !faceMode;
    microphoneButton.hidden = inputMethod !== "speech";
    analyseButton.textContent = faceMode
      ? "Analyse and save facial expression →"
      : "Analyse and save my mood →";
    errorBox.hidden = true;
    if (inputMethod !== "speech" && recognition && listening) recognition.stop();
    if (!faceMode) stopCamera();
  };

  modeButtons.forEach((button) => button.addEventListener("click", () => updateMode(button)));

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = "en-NG";
    recognition.continuous = false;
    recognition.interimResults = true;
    let baseText = "";
    recognition.onstart = () => {
      listening = true;
      baseText = textArea.value.trim();
      microphoneButton.textContent = "■ Stop listening";
      microphoneButton.classList.add("recording");
    };
    recognition.onresult = (event) => {
      let transcript = "";
      for (let index = 0; index < event.results.length; index += 1) {
        transcript += event.results[index][0].transcript;
      }
      textArea.value = `${baseText}${baseText ? " " : ""}${transcript}`.slice(0, 2000);
      updateCount();
    };
    recognition.onend = () => {
      listening = false;
      microphoneButton.textContent = "● Start speaking";
      microphoneButton.classList.remove("recording");
    };
    recognition.onerror = () => showError(
      "Speech could not be captured. Check microphone permission or type the entry."
    );
    microphoneButton.addEventListener("click", () => (
      listening ? recognition.stop() : recognition.start()
    ));
  } else {
    microphoneButton.addEventListener("click", () => showError(
      "Speech input is unavailable in this browser. Use Chrome or Edge, or type the entry."
    ));
  }

  startCameraButton.addEventListener("click", async () => {
    errorBox.hidden = true;
    if (cameraStream) {
      stopCamera();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("Camera access is unavailable in this browser. Upload a photo instead.");
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
        audio: false,
      });
      cameraVideo.srcObject = cameraStream;
      cameraVideo.hidden = false;
      facePreview.hidden = true;
      cameraPlaceholder.hidden = true;
      captureButton.hidden = false;
      startCameraButton.textContent = "■ Stop camera";
      await cameraVideo.play();
    } catch (_error) {
      stopCamera();
      showError("Camera permission was denied or the camera is unavailable. Upload a photo instead.");
    }
  });

  captureButton.addEventListener("click", () => {
    if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
      showError("The camera is not ready yet. Wait a moment and try again.");
      return;
    }
    const scale = Math.min(1, 1024 / cameraVideo.videoWidth);
    cameraCanvas.width = Math.round(cameraVideo.videoWidth * scale);
    cameraCanvas.height = Math.round(cameraVideo.videoHeight * scale);
    const context = cameraCanvas.getContext("2d");
    context.translate(cameraCanvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    context.setTransform(1, 0, 0, 1, 0, 0);
    faceImageData = cameraCanvas.toDataURL("image/jpeg", 0.88);
    faceSource = "camera";
    facePreview.src = faceImageData;
    facePreview.hidden = false;
    stopCamera();
    cameraPlaceholder.hidden = true;
    errorBox.hidden = true;
  });

  faceFile.addEventListener("change", () => {
    const file = faceFile.files?.[0];
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      showError("Choose a JPG, PNG, or WEBP photo.");
      faceFile.value = "";
      return;
    }
    if (file.size > 3 * 1024 * 1024) {
      showError("The selected photo is larger than 3 MB.");
      faceFile.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      stopCamera();
      faceImageData = String(reader.result || "");
      faceSource = "upload";
      facePreview.src = faceImageData;
      facePreview.hidden = false;
      cameraPlaceholder.hidden = true;
      errorBox.hidden = true;
    };
    reader.onerror = () => showError("The selected photo could not be read.");
    reader.readAsDataURL(file);
  });

  const make = (tag, className, content) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = content;
    return element;
  };

  const renderResult = (data) => {
    document.getElementById("emptyResult").hidden = true;
    document.getElementById("resultContent").hidden = false;
    document.getElementById("resultCard").classList.remove("empty");
    document.getElementById("resultEmoji").textContent = data.details.emoji;
    document.getElementById("resultLead").textContent = data.modality === "facial expression"
      ? "Visible expression resembles"
      : "Your words feel";
    const moodHeading = document.getElementById("resultMood");
    moodHeading.textContent = data.details.label;
    moodHeading.style.color = data.details.colour;
    document.getElementById("resultConfidence").textContent = (
      `${data.confidence.toFixed(1)}% model confidence · ${data.model_name}`
    );
    document.getElementById("resultDescription").textContent = data.details.description;
    document.getElementById("resultSuggestion").textContent = data.details.suggestion;
    document.getElementById("urgentSupport").hidden = !data.needs_urgent_support;

    const signalList = document.getElementById("signalList");
    signalList.replaceChildren();
    (data.signals.length ? data.signals : ["No single strong signal"]).forEach((signal) => (
      signalList.append(make("b", "", signal))
    ));

    const probabilityList = document.getElementById("probabilityList");
    probabilityList.replaceChildren();
    data.probabilities.forEach((item) => {
      const row = make("div", "probability-row");
      row.append(make("span", "", item.label));
      const track = make("i");
      const fill = make("b");
      fill.style.width = `${Math.max(item.probability, 1)}%`;
      fill.style.background = item.colour;
      track.append(fill);
      row.append(track, make("strong", "", `${item.probability.toFixed(1)}%`));
      probabilityList.append(row);
    });
    document.getElementById("resultCard").scrollIntoView({ behavior: "smooth", block: "center" });
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    const faceMode = inputMethod === "face";
    const text = textArea.value.trim();
    if (faceMode && !faceImageData) {
      showError("Capture a webcam photo or upload an image before analysing.");
      return;
    }
    if (!faceMode && text.length < 3) {
      showError("Please write or speak a little more before analysing.");
      return;
    }
    analyseButton.disabled = true;
    analyseButton.textContent = faceMode ? "Reading the facial pattern…" : "Reading your words…";
    try {
      const response = await fetch(faceMode ? "/api/analyze-face" : "/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
        body: JSON.stringify(faceMode
          ? { image: faceImageData, source: faceSource }
          : { text, input_method: inputMethod }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Mood analysis failed.");
      renderResult(data);
    } catch (error) {
      showError(error.message);
    } finally {
      analyseButton.disabled = false;
      analyseButton.textContent = faceMode
        ? "Analyse and save facial expression →"
        : "Analyse and save my mood →";
    }
  });

  window.addEventListener("beforeunload", stopCamera);
})();
