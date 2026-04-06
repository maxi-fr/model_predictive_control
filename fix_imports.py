with open("src/model_predictive_control/ocp.py") as f:
    content = f.read()

if "import warnings" not in content:
    content = "import warnings\n" + content

if "from collections.abc import Callable" not in content:
    content = "from collections.abc import Callable\n" + content

with open("src/model_predictive_control/ocp.py", "w") as f:
    f.write(content)
