from nox_poetry import session


@session(tags=["style", "fix"])
def black(session):
    session.install("pytest", ".")
    session.run("pytest")


@session(tags=["style", "fix"])
def isort(session):
    session.install("isort")
    session.run("isort", ".")


@session(tags=["style"])
def flake8(session):
    session.install("flake8")
    session.run("flake8", ".")


@session(python="3.9")
def tests(session):
    session.install("pytest", ".")
    session.run("pytest")
