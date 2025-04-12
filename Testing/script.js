import testdata from './data.json' with { type: 'json' };
let nn
let i = 0;
let correct = 0;
const labelStats = {}; // { label: { total: 0, correct: 0 } }

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

    nn.load(options, startTesting)
}

function showLabel(){
    console.log('ready to test');
    for (let testpose of testdata) {
        console.log('Label:', testpose.label)
    }
}

function startTesting() {
    let testpose = testdata[i];

    nn.classify(testpose.data, (results) => {
        const predicted = results[0].label;
        const actual = testpose.label;

        console.log(`I think it's a: ${predicted}, but it's actually: ${actual}`);

        if (!labelStats[actual]) {
            labelStats[actual] = { total: 0, correct: 0 };
        }
        labelStats[actual].total++;

        if (predicted === actual) {
            correct++;
            labelStats[actual].correct++;
        }

        i++;
        if (i < testdata.length) {
            startTesting();
        } else {
            // Totale accuracy
            const accuracy = (correct / testdata.length) * 100;
            console.log(`\nTotal accuracy: ${correct}/${testdata.length} (${accuracy.toFixed()}%)\n`);

            // Accuracy per label
            for (const label in labelStats) {
                const { total, correct } = labelStats[label];
                const labelAccuracy = (correct / total) * 100;
                console.log(`Label (${label}): ${correct}/${total} correct (${labelAccuracy.toFixed()}%)`);
            }
        }
    });
}

showLabel()
createNeuralNetwork()