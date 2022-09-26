I=imread("C:\Users\fasi\OneDrive\Desktop\image2.jpg")
imshow(I)
[x,y]=ginput(2)
z=1016
fx=1235.1912
fy=1232.9848
x1=z*(x(1)/fx)
x2=z*(x(2)/fx)
y1=z*(y(1)/fy)
y2=z*(y(2)/fy)
dist=sqrt((y2-y1)^2+(x2-x1)^2)
fprintf("the distance",dist)