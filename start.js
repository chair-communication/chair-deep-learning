const tf = require("@tensorflow/tfjs");
const split = require("./split");
const insert_words = require("./switch");
const model_initialize = require("./model_initialize");
const fs = require("fs");
require("@tensorflow/tfjs-node");
async function main() {
  var loadedModel = model_initialize();
  var model = model_initialize();

  const input_data = await split("./dataset/d_120.csv");
  const input_test = [];
  const label_test = [];
  input_data.forEach((el) => {
    input_test.push(el.splice(0, 35));
    label_test.push(el);
  });
  const x_realtime = tf.tensor2d(input_test);
  const y_realtime = tf.tensor2d(label_test);

  try {
    loadedModel = await tf.loadLayersModel(
      "file://./model_0602_epochs_60/model.json"
    );
  } catch (e) {
    console.log("불러오기 실패");
  }

  console.log("==============================================");
  // const predictions = loadedModel.predict(x_realtime, 10).argMax(1);
  const predictions = loadedModel.predict(x_realtime, 10);
  var tmp = predictions.dataSync();
  var p = predictions.dataSync();
  const label = y_realtime.argMax(1);
  var l = label.dataSync();
  predict = JSON.stringify(predictions);
  // tmp = JSON.stringify(tmp);
  result = JSON.stringify(label);

  fs.writeFile("./predictions.txt", predict, () => {});
  fs.writeFile("./result.txt", result, () => {});

  for (i in p) {
    console.log(tmp[i]);
  }
  // for (i in p) {
  //   console.log(`예측: ${insert_words(p[i])}`);
  //   console.log(`실제: ${insert_words(l[i])}`);
  // }
  // console.log(`실제 자세는 ${insert_words(l[0])}입니다`);
}
main();
