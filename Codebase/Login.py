def login(username, password):
    if username == "" or password == "":
        return "Missing credentials"
    if username == "admin" and password == "admin":
        return "Login successful"
    return "Invalid credentials"
