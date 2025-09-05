import typer
from ysautoml.data.fyi.cli import main_dsa, main_dm
from ysautoml.data.dsbn.cli import main as main_dsbn

app = typer.Typer(help="YS-AutoML: Unified CVLab utilities")

# Data utilities
app.command("dsa")(main_dsa)
app.command("dm")(main_dm)

app.command("dsbn")(main_dsbn)

# Optimization utilities (추가 예정)
# app.command("fxp")(fxp_cli)
# app.command("loss-search")(loss_search_cli)
# app.command("mtl-loss")(mtl_loss_cli)

# Network utilities (추가 예정)
# app.command("fewshot")(fewshot_cli)
# app.command("zeroshot")(zeroshot_cli)
# app.command("oneshot")(oneshot_cli)

if __name__ == "__main__":
    app()
