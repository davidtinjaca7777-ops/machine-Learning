from flask import request, render_template
import RLModel as rl


def init_rl_routes(app):
    @app.route("/rl_concepts")
    def rl_concepts():
        return render_template("rl_concepts.html")

    @app.route("/rl_app", methods=["GET", "POST"])
    def rl_app():
        # Trains only when "Train Agent" is pressed (POST)
        result = rl.train_and_evaluate() if request.method == "POST" else None
        return render_template("rl_app.html", grid=rl.GRID, counts=rl.counts(), rewards=rl.REWARDS,
                               cfg=rl.CONFIG, result=result, actions=rl.ACTIONS)
