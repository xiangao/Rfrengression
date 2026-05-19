# Package index

## Model constructors

- [`frengression()`](https://xiangao.github.io/Rfrengression/reference/frengression.md)
  : Create a Frengression Model
- [`frengression_seq()`](https://xiangao.github.io/Rfrengression/reference/frengression_seq.md)
  : Create a FrengressionSeq Model for Longitudinal Data
- [`frengression_surv()`](https://xiangao.github.io/Rfrengression/reference/frengression_surv.md)
  : Create a FrengressionSurv Model for Survival Data

## Training and causal specification

- [`train_y()`](https://xiangao.github.io/Rfrengression/reference/train_y.md)
  : Train the Outcome Model (Y)
- [`train_xz()`](https://xiangao.github.io/Rfrengression/reference/train_xz.md)
  : Train the Marginal Model (XZ)
- [`train_e()`](https://xiangao.github.io/Rfrengression/reference/train_e.md)
  : Train the Confounder Model (E)
- [`specify_causal()`](https://xiangao.github.io/Rfrengression/reference/specify_causal.md)
  : Specify a Custom Causal Margin
- [`reset_y_models()`](https://xiangao.github.io/Rfrengression/reference/reset_y_models.md)
  : Reset Outcome Models

## Prediction and sampling

- [`predict_causal()`](https://xiangao.github.io/Rfrengression/reference/predict_causal.md)
  : Predict Causal Effects (generic)
- [`sample_joint()`](https://xiangao.github.io/Rfrengression/reference/sample_joint.md)
  : Sample from the Joint Distribution
- [`sample_xz()`](https://xiangao.github.io/Rfrengression/reference/sample_xz.md)
  : Sample from the Marginal Distribution of (X, Z)
- [`sample_causal_margin()`](https://xiangao.github.io/Rfrengression/reference/sample_causal_margin.md)
  : Sample from the Causal Margin P(Y \| do(X))

## Methods

- [`predict(`*`<frengression_seq>`*`)`](https://xiangao.github.io/Rfrengression/reference/predict.frengression.md)
  [`predict(`*`<frengression_surv>`*`)`](https://xiangao.github.io/Rfrengression/reference/predict.frengression.md)
  [`predict(`*`<frengression>`*`)`](https://xiangao.github.io/Rfrengression/reference/predict.frengression.md)
  : Predict Causal Effects
- [`print(`*`<frengression>`*`)`](https://xiangao.github.io/Rfrengression/reference/print.frengression.md)
  : Print a Frengression Model
- [`print(`*`<frengression_seq>`*`)`](https://xiangao.github.io/Rfrengression/reference/print.frengression_seq.md)
  : Print a FrengressionSeq Model
- [`print(`*`<frengression_surv>`*`)`](https://xiangao.github.io/Rfrengression/reference/print.frengression_surv.md)
  : Print a FrengressionSurv Model
