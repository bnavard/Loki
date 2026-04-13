# Tests
TODO: test might not be updated with the most recent state of the code!

## test_pipeline.py
Unit tests with random tensors (no data required). Verifies shapes through THConditioning, AudioEncoder, THUnetModel, THDiffusion, and THSampler.

```bash
PYTHONPATH=. python marionette/tests/test_pipeline.py
```

## test_training_integration.py
End-to-end test with 2 real clips: dataset → VAE → conditioning → UNet → loss → backward.

```bash
PYTHONPATH=. python marionette/tests/test_training_integration.py
```

## test_dataset.py
Visual dataset test. Saves comparison images to `outputs/dataset_vis/`.

```bash
python marionette/tests/test_dataset.py
```
