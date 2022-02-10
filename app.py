from flask import Flask, redirect, url_for, render_template, request, session, flash
import os
import random
import re
from datetime import timedelta


class Base(object):
    def __init__(self):
        pass

    def build(self, img_dir):
        raise NotImplementedError("Not yet build data split!")


class Database(Base):
    def __init__(self, root_dir='static', ann_dir='anns'):
        super(Database, self).__init__()
        self.root_dir = root_dir
        self.ann_dir = ann_dir
        self.imgs = None
        self.img_dir = None
        self.index = 0

    def build(self, img_dir):
        if not os.path.exists(os.path.join(self.root_dir, img_dir)):
            raise ValueError("Database split does not exist!")
        elif not os.listdir(os.path.join(self.root_dir, img_dir)):
            raise ValueError("Split folder does not contain sources!")

        self.img_dir = img_dir
        self.imgs = os.listdir(os.path.join(self.root_dir, self.img_dir))
        self.index = 0

    def get_img(self):
        if self.index >= len(self.imgs):
            self.index = 0
            return None, self.index
        else:
            img = self.imgs[self.index]
            return os.path.join(self.img_dir, img), self.index

    def _set_success(self):
        self.index += 1

    def set_ann(self, cap, username=''):
        if not os.path.exists(os.path.join(self.root_dir, self.ann_dir)):
            os.mkdir(os.path.join(self.root_dir, self.ann_dir))

        my_ann = os.path.join(self.root_dir, self.ann_dir, self.img_dir + '-' + username + '.txt')

        with open(my_ann, 'a+') as f:
            if self.index == 0:
                f.write("Username: " + username + '\n')
            f.write(os.path.join(self.img_dir, self.imgs[self.index]) + ': ' + cap + '\n')
            self._set_success()


db = Database()

regex = "^[A-Za-z0-9_]\\w+$"
NAMEPATTERN = re.compile(regex)

app = Flask(__name__)
app.secret_key = "012345678"
app.permanent_session_lifetime = timedelta(minutes=120)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        username = request.form["nm"]
        split_choice = request.form["submit"]
        if re.fullmatch(NAMEPATTERN, username):
            session.permanent = True
            session["nm"] = username
            session["split"] = split_choice
            db.build(img_dir=session["split"])
            return redirect(url_for("survey", usrnm=username))
        else:
            flash("Username should be at least 2 chars or numbers without space!", "info")
            return render_template("home.html")
    else:
        return render_template("home.html")


@app.route("/survey/", methods=["POST", "GET"])
def survey(usrnm='unknown user'):
    if "nm" in session:
        #if request.args.get('usrnm') is not '':
        #    usrnm = request.args.get('usrnm')
        usrnm = session["nm"]

        if request.method == "POST":
            caption = request.form["caption"]
            if caption is not None and caption != '':
                db.set_ann(caption, usrnm)
                return redirect(url_for("survey", usrnm=usrnm))
            else:
                flash("Please type in a valid caption.", 'info')
                return redirect(url_for("survey", usrnm=usrnm))
        else:
            img, idx = db.get_img()
            if img is None:
                flash("You have completed this survey. Thank you!", "info")
                return redirect(url_for("home"))
            return render_template("survey.html", name=usrnm, imgs=[img], idx=idx)
    else:
        flash("Please type in your username.", "info")
        return redirect(url_for("home"))


@app.route("/logout/")
def logout():
    if "nm" in session:
        usrnm = session["nm"]
        flash(f"You have logout, {usrnm}.", "info")
    session.pop("nm", None)
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)
