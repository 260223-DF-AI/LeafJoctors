# LeafJoctors

## Datasets
- [OLID Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523147/#s6) - Gazipur, Bangladesh
- [Plant Pathology Dataset](https://data.mendeley.com/datasets/hb74ynkjcn/1) - Katra, India


## Process to run program
*Run following commands from the root directory*
- Ensure model is deployed to SageMaker: run `py sage/deploy.py` to deploy model
- Boot FastAPI server with `py -m app`
- Access website by opening `index.html`

- Run model training program with `py -m train_model.resnet`