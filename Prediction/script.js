import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton")
const logButton = document.getElementById("logButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")

let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let nn
let statusDiv = document.getElementById("statusDiv")

function createNeuralNetwork() {
    ml5.setBackend("webgl")
    nn = ml5.neuralNetwork({
        task: 'classification',
        debug: true,
    });

    const options = {
        model: "model/model.json",
        metadata: "model/model_meta.json",
        weights: "model/model.weights.bin",
    }

    nn.load(options, createHandLandmarker)
}

/********************************************************************
// CREATE THE POSE DETECTOR
********************************************************************/
const createHandLandmarker = async () => {
    console.log("Neural Network loaded")
    
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    logButton.addEventListener("click", (e) => classifyHand(e))
}

/********************************************************************
// START THE WEBCAM
********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            video.style.display = "block"; 
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
        enableWebcamButton.style.display = "none";
        logButton.style.display = "block";
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
// START PREDICTIONS    
********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}

/********************************************************************
// QUIZ
********************************************************************/
let quizIndex = 0;
const quizQuestions = [
    { question: "Wat is 0+1?", correctAnswer: "1" },
    { question: "Wat is 1 + 2?", correctAnswer: "3" },   
    { question: "Wat is 2 + 0?", correctAnswer: "2" }, 
    { question: "Wat is 5 - 4?", correctAnswer: "1" },
    { question: "Wat is 9 - 6?", correctAnswer: "3" },
    { question: "Wat is 2x1?", correctAnswer: "2" },
];

function updateQuizQuestion() {
    if (quizIndex < quizQuestions.length) {
        document.getElementById("question").innerText = quizQuestions[quizIndex].question;
    }
}

function classifyHand() {
    if (results.landmarks.length === 0) {
        statusDiv.innerText = "Geen hand gevonden, probeer opnieuw";
        return;
    }

    let numbersOnly = [];
    let hand = results.landmarks[0];

    for (let point of hand) {
        numbersOnly.push(point.x, point.y, point.z);
    }

    nn.classify(numbersOnly, (results) => {
        const label = results[0].label;
        const confidence = (results[0].confidence.toFixed(2)) * 100;
        statusDiv.innerText = `Jij koos nummer: ${label}`;
        console.log(`Jij koos nummer: ${label} (Zekerheid: ${confidence}%)`)

        checkAnswer(label);
    });
}

function checkAnswer(userAnswer) {
    const correct = quizQuestions[quizIndex].correctAnswer;

    if (userAnswer === correct) {
        statusDiv.innerText += "\n✅ Goed gedaan!";
        quizIndex++;
        updateQuizQuestion()
    } else {
        statusDiv.innerText += "\n❌ Helaas, probeer opnieuw!";
    }
}

updateQuizQuestion();
createNeuralNetwork()