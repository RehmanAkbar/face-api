const save = require("./saveFile");
const path = require("path");

const tf = require("@tensorflow/tfjs-node");

const canvas = require("canvas");

const faceapi = require("@vladmandic/face-api/dist/face-api.node.js");
const modelPathRoot = "../models";

let optionsSSDMobileNet;

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function image(file) {
  const decoded = tf.node.decodeImage(file);
  const casted = decoded.toFloat();
  const result = casted.expandDims(0);
  decoded.dispose();
  casted.dispose();
  return result;
}


async function detect(tensor) {
  const result = await faceapi.detectAllFaces(tensor, optionsSSDMobileNet);
  return result;
}


async function main(file, filename) {


  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready();

  console.log(
    `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${faceapi.version.faceapi
    } Backend: ${faceapi.tf?.getBackend()}`
  );

  console.log("Loading FaceAPI models");
  const modelPath = path.join(__dirname, modelPathRoot);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);


  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
    minConfidence: 0.5,
  });

  const tensor = await image(file);
  const result = await detect(tensor);
  console.log("Detected faces:", result.length);

  const canvasImg = await canvas.loadImage(file);
  const out = await faceapi.createCanvasFromMedia(canvasImg);
  faceapi.draw.drawDetections(out, result);
  save.saveFile(filename, out.toBuffer("image/jpeg"));
  console.log(`done, saved results to ${filename}`);

  tensor.dispose();

  return result;
}


async function recognition(files) {


  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready();

  // console.log(
  //   `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${faceapi.version.faceapi
  //   } Backend: ${faceapi.tf?.getBackend()}`
  // );

  // console.log("Loading FaceAPI models");
  const modelPath = path.join(__dirname, modelPathRoot);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath)
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath)

  const REFERENCE_IMAGE = files.reference.data
  const QUERY_IMAGE = files.query.data

  const referenceImage = await canvas.loadImage(REFERENCE_IMAGE)
  const queryImage = await canvas.loadImage(QUERY_IMAGE)

  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
    minConfidence: 0.5,
  });

  const resultsRef = await faceapi.detectAllFaces(referenceImage, optionsSSDMobileNet)
    .withFaceLandmarks()
    .withFaceDescriptors()

  const resultsQuery = await faceapi.detectAllFaces(queryImage, optionsSSDMobileNet)
    .withFaceLandmarks()
    .withFaceDescriptors()

  if (!resultsRef.length) {
    return ['no face detected'];
  }
  if (!resultsQuery.length) {
    return ['no face detected'];
  }

  const faceMatcher = new faceapi.FaceMatcher(resultsRef)

  const labels = faceMatcher.labeledDescriptors.map(ld => ld.label)
  // const refDrawBoxes = resultsRef.map(res => res.detection.box).map((box, i) => new faceapi.draw.DrawBox(box, { label: labels[i] }))
  // const outRef = faceapi.createCanvasFromMedia(referenceImage)
  // refDrawBoxes.forEach(drawBox => drawBox.draw(outRef))
  const queryDrawBoxes = resultsQuery.map(res => {
    const bestMatch = faceMatcher.findBestMatch(res.descriptor)
    // return res.detection.box, { label: bestMatch.toString() }
    return { box: res.detection.box, label: bestMatch.toString() };
    return new faceapi.draw.DrawBox(res.detection.box, { label: bestMatch.toString() })
  })
  return queryDrawBoxes;
  // console.log(queryDrawBoxes)
  // console.log(resultsRef);
  // return resultsRef;
}

module.exports = {
  detect: main,
  recognition: recognition
};
