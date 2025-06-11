const X = labels.map((_, i) => [
  btcReturns[i] ?? 0,
  m2Growth[i] ?? 0,
  inflacao[i] ?? 0, // inflação mensal (você pode usar IPCA)
  selic[i] ?? 0      // taxa Selic mensal
]);

const y = btcReturns.map(r => r > 0 ? 1 : 0); 

    import * as tf from '@tensorflow/tfjs';

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, inputShape: [X[0].length], activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return model;
}

async function trainEnsemble(X, y, ensembleSize = 5) {
  const models = [];
  const xs = tf.tensor2d(X);
  const ys = tf.tensor2d(y, [y.length, 1]);

  for (let i = 0; i < ensembleSize; i++) {
    const model = createModel();
    await model.fit(xs, ys, { epochs: 50, verbose: 0 });
    models.push(model);
  }

  return models;
}

async function predictWithEnsemble(models, input) {
  const inputTensor = tf.tensor2d([input]);
  const predictions = await Promise.all(models.map(m => m.predict(inputTensor).data()));
  const avg = predictions.reduce((sum, p) => sum + p[0], 0) / models.length;
  return avg > 0.5 ? 1 : 0;
}
