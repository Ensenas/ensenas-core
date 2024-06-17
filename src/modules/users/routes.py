from flask import jsonify, request
from . import product_blueprint
from ...models.product import Product
from ... import db

@product_blueprint.route('/', methods=['GET'])
def get_users():
    products = Product.query.all()
    return jsonify([product.to_dict() for product in products])