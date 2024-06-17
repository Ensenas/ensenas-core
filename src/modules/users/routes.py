from flask import Blueprint, request, jsonify
from ...extensions import db
from ...models.user import User
from ...schemas.user import UserSchema

user_blueprint = Blueprint('user', __name__)
product_schema = ProductSchema()
products_schema = ProductSchema(many=True)

@user_blueprint.route('/', methods=['POST'])
def add_product():
    data = request.get_json()
    try:
        product = product_schema.load(data)
    except ValidationError as err:
        return jsonify(err.messages), 400

    db.session.add(product)
    db.session.commit()

    return product_schema.dump(product), 201

@product_blueprint.route('/', methods=['GET'])
def get_products():
    products = Product.query.all()
    return jsonify(products_schema.dump(products))