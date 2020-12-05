from app import db
from sqlalchemy.dialects.postgresql import JSON

# class Result(db.Model):
#     __tablename__ = 'wordcount_dev'

#     id = db.Column(db.Integer, primary_key=True)
#     url = db.Column(db.String())
#     result_all = db.Column(JSON)
#     result_no_stop_words = db.Column(JSON)

#     def __init__(self, url, result_all, result_no_stop_words):
#         self.url = url
#         self.result_all = result_all
#         self.result_no_stop_words = result_no_stop_words

#     def __repr__(self):
#         return '<id {}>'.format(self.id)

class Word2Vec(db.Model):
    __tablename__ = "cord19_2020_vecs"

    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String())
    vector = db.Column(db.ARRAY(db.REAL))

    def __repr__(self):
        return "<Word2Vec(id='{}', word='{}', vector={})>"\
            .format(self.id, self.word, self.vector)

# class Candidate(db.Model):
#     __tablename__ = "top_authors"

#     id = db.Column(db.Integer, primary_key=True)
#     author_id = db.Column(db.String())
#     vector = db.Column(db.ARRAY(db.REAL))

#     def __repr__(self):
#         return "<Candidate(id='{}', author_id='{}', vector={})>"\
#             .format(self.id, self.author_id, self.vector)

class CandidateInfo(db.Model):
    __tablename__ = "top_authors_info"

    id = db.Column(db.Integer, primary_key=True)
    author_id = db.Column(db.String())
    author_name = db.Column(db.String())
    author_affiliation = db.Column(db.String())

    def __repr__(self):
        return "<CandidateInfo(id='{}', author_id='{}', author_name='{}', author_affiliation='{}')>"\
            .format(self.id, self.author_id, self.author_name, self.author_affiliation)

class Candidate(db.Model):
    __tablename__ = "malaysia_author_2018"

    id = db.Column(db.Integer, primary_key=True)
    author_ids = db.Column(db.String())
    author_names = db.Column(db.String())
    affilname = db.Column(db.String())
    description_vector = db.Column(db.ARRAY(db.Float))

    def __repr__(self):
        return "<Candidate(id='{}', author_ids='{}', author_names='{}', author_names='{}', description_vector={})>"\
            .format(self.id, self.author_ids, self.author_names, self.affilname, self.description_vector)