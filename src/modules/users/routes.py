from flask import Blueprint, request, jsonify
from ...extensions import db
from ...models.user import User
from ...schemas.user import UserSchema

user_blueprint = Blueprint('user', __name__)
user_schema = UserSchema()
users_schema = UserSchema(many=True)

@user_blueprint.route('/', methods=['POST'])
def add_user():
    data = request.get_json()
    try:
        user = user_schema.load(data)
    except ValidationError as err:
        return jsonify(err.messages), 400

    db.session.add(user)
    db.session.commit()

    return users_schema.dump(user), 201

@user_blueprint.route('/', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify(users_schema.dump(users))