#FENtoVEC function from https://www.kaggle.com/code/gabrielhaselhurst/chess-dataset/notebook
# Outputs array of size 64 (NEED TO CONVERT TO 8x8)
def FENtoVEC (FEN):
    pieces = {"r":5,"n":3,"b":3.5,"q":9.5,"k":20,"p":1,"R":-5,"N":-3,"B":-3.5,"Q":-9.5,"K":-20,"P":-1}
    FEN = list(str(FEN.split()[0]))
    VEC = []
    for i in range(len(FEN)):
        if FEN[i] == "/":
            continue
        if FEN[i] in pieces:
            VEC.append(pieces[FEN[i]])
        else:
            em = [VEC.append(0) for i in range(int(FEN[i]))]

    return VEC