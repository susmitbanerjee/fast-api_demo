from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import os

from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

app = FastAPI()

# Mount the static directory to serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")


def execute_notebook(notebook_path, csv_path):
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Set the input CSV file path in the notebook
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace("input_csv_path = 'Position_Salaries.csv'",
                                              f"input_csv_path = '{csv_path}'")

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})

    return nb


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        contents = await file.read()
        tmp.write(contents)
        csv_path = tmp.name

    # Path to your Jupyter Notebook
    notebook_path = 'Simple_Linear_Regression.ipynb'

    # Execute the notebook
    executed_nb = execute_notebook(notebook_path, csv_path)

    # Extract and save the generated plots
    plot_paths = []
    for cell in executed_nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output and 'image/png' in output['data']:
                    plot_path = os.path.join(tempfile.gettempdir(), f"plot_{len(plot_paths)}.png")
                    with open(plot_path, 'wb') as f:
                        f.write(output['data']['image/png'])
                    plot_paths.append(plot_path)

    if not plot_paths:
        raise HTTPException(status_code=400, detail="No plots generated from the notebook.")

    return {"plot_paths": plot_paths}
