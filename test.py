from PIL import Image
 
 
def cut(file_name):
    im = Image.open(file_name)
    newim = im.convert("RGB")
 
    source = im.split()
    R, G, B = 0, 1, 2  
    rmask = source[R].point(lambda i:(i >= 255 and 255))#如果大于255，就赋值255，如果小于255，就赋值0
    gmask = source[G].point(lambda i:(i >= 150 and 255))
    bmask = source[B].point(lambda i:(i >= 255 and 255))
 
    out = Image.new("RGBA", im.size, None)
    # out是一个所有像素点都是0000的图片，然后根据前面的rmask,把要屏蔽的地方mask255,所以就都变成黑色了啦
    newim.paste(out, None, rmask)
    newim.paste(out, None, gmask)
    # newim.paste(out, None, bmask)      
 
    out.save('out.png')
    newim.save("new_" + file_name)
 
cut('2.jpg')