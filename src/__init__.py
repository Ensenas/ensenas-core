from flask import Flask
from .models.users import users_blueprint
from .extensions import init_extensions

def create_app(config_name):
    app = Flask(__name__)
    
    # Configurations
    app.config.from_object(config_name)

    # Initialize extensions
    init_extensions(app)

    # db.init_app(app)
    # migrate.init_app(app, db)

    # Register Blueprints
    app.register_blueprint(users_blueprint, url_prefix='/users')
