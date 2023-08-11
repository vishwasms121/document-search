from fileinput import filename
from flask_sqlalchemy import SQLAlchemy
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.sql import func


db = SQLAlchemy()


@dataclass
class AllFiles(db.Model):
    __tablename__ = "all_files"
    id: int = db.Column(db.BigInteger, primary_key=True)
    filename: str = db.Column(db.String(100))
    created_date: datetime = db.Column(db.Date)

    def __init__(self, filename, created_date):
        self.filename = filename
        self.created_date = created_date

    def __repr__(self):
        return "AllFiles <filename=%s>" % (
            self.filename
        )

@dataclass
class MasterExtraction(db.Model):
    __tablename__ = "master_extraction"
    id: int = db.Column(db.BigInteger, primary_key=True)
    file_id: int = db.Column(db.BigInteger, db.ForeignKey("all_files.id"))
    page_num: int = db.Column(db.BigInteger)
    para_num: int = db.Column(db.BigInteger)
    extracted_text: str = db.Column(db.String(5000))
    cleaned_extracted_text: str = db.Column(db.String(5000))
    created_date: datetime = db.Column(db.Date)

    def __init__(self, file_id,page_num,para_num,
                 extracted_text ,cleaned_extract_text, created_date):
        self.file_id = file_id
        self.page_num = page_num
        self.para_num = para_num
        self.extracted_text = extracted_text
        self.cleaned_extracted_text = cleaned_extract_text
        self.created_date = created_date

    def __repr__(self):
        return "MasterExtraction <extracted_text=%s>" % (
            self.extracted_text
        )