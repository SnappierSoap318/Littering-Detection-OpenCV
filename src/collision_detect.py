#collision detection between 2 boxes 
def collision(box1, box2):
    if box1.x < box2.w and box1.w > box2.x and box1.y < box2.h and box1.h > box2.y:
        return True
    else:
        return False
