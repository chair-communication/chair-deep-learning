const tf = require("@tensorflow/tfjs");
const split = require("./split");
const model_initialize = require("./model_initialize");
const insert_words = require("./switch");
require("@tensorflow/tfjs-node");
const { loadGraphModel } = require("@tensorflow/tfjs-converter");

function onBatchEnd(batch, logs) {
  console.log("Accuracy", logs.acc);
}
async function main() {
  model = model_initialize();
  const data = await split(
    "C:/Users/park ji seong/Desktop/데이터셋/0602_dataset_with_today_data.csv"
  );
  const input_data = [];
  const label_data = [];
  data.forEach((el) => {
    input_data.push(el.splice(0, 35));
    label_data.push(el);
  });
  const input = tf.tensor2d(input_data);
  const label = tf.tensor2d(label_data);
  await model.fit(input, label, {
    epochs: 80,
    callbacks: {
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch ${epoch}: loss = ${log.loss.toString()}`),
    },
  });
  try {
    const saveResults = await model.save("file://./model_0602_epochs_30/");
    console.log(saveResults);
  } catch (e) {
    console.log("저장 실패");
  }
}

main();
