import posedata from './data.json' with { type: 'json' };

const statusDiv = document.getElementById('status');
let nn

function startTraining() {
    statusDiv.textContent = 'Training Neural Network';

    ml5.setBackend('webgl');
    nn = ml5.neuralNetwork({
        task: 'classification',
        debug: true,
    });

console.log(`adding ${posedata.length} poses`)

    for(let pose of posedata) {
        nn.addData(pose.data, { label: pose.label });
    }

    nn.normalizeData();
    nn.train({ epochs: 150 }, finishedTraining);
}

for (let pose of posedata) {
    console.log(`Adding: ${pose.label}`, pose.data.length);
}

function finishedTraining() {
    statusDiv.textContent = 'Training finished!';
    nn.save()
}

startTraining();