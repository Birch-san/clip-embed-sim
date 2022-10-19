### Setup

#### (Optional) make & source a virtualenv

```bash
python -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Get data

Create a directory `square` in the root of the repository with 1:1 aspect ratio images.  
Create a file `img_to_caption.json` providing a mapping of image filename to text caption for every image you placed in `square`. An example, `img_to_caption.example.json` is provided to demonstrate the expected schema.

### Run

```bash
python embed-sim.py
```