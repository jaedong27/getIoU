import numpy as np
import cv2

def dist(p1, p2):
    return (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1])

def ccw(p1, p2, p3):
    cross_product = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p3[0] - p1[0])*(p2[1] - p1[1])
    if (cross_product > 0):
        return 1
    elif (cross_product < 0):
        return -1
    else:
        return 0

def SortComparator(p, left, right):
    ret = 0
    direction = ccw(p, left, right)
    if(direction == 0):
        ret = (dist(p, left) < dist(p, right))
    elif(direction == 1):
        ret = 1
    else:
        ret = 0
    return ret

def BubbleSort(a):
    p = a[0,:]

    length = a.shape[0]
    
    for i in range(1, length-1):
        for j in range(i, length-1):
            # i, j Compare
            print(i,j,SortComparator(p, a[j,:], a[j+1,:]))
            if SortComparator(p, a[j,:], a[j+1,:]) <= 0:
                #print("a:", a[i,:], a[j,:])
                temp = np.copy(a[j, :])
                a[j, :] = np.copy(a[j+1, :])
                a[j+1, :] = temp
                #print("b:", a[i,:], a[j,:], temp)
                #print(i,j)

    return a

# (p1, p2)를 이은 직선과 (p3, p4)를 이은 직선의 교차점을 구하는 함수
# Function to get intersection point with line connecting points (p1, p2) and another line (p3, p4).
def IntersectionPoint(p1, p2, p3, p4):
    ret = np.array([0.0,0.0])

    ret[0] = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0]))/( (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]) )

    ret[1] = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / ( (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]) )

    return ret


# 다각형의 넓이를 구한다.
# find the area of a polygon
def GetPolygonArea(points):
        num_points = points.shape[0]
        i = num_points - 1
        ret = 0
        for j in range(num_points):
            ret += points[i,0] * points[j,1] - points[j,0] * points[i,1]
            i = j

        #ret = ret < 0 ? -ret : ret
        ret /= 2

        return abs(ret)

#// 두 선분이 교차하면 1을 교차하지 않으면 0을 반환합니다.
#// If the two segments intersect, they will return 1 if they do not intersect 0.
def LineIntersection(l1_p1, l1_p2, l2_p1, l2_p2):
    
    #// l1을 기준으로 l2가 교차하는 지 확인한다.
    #// Check if l2 intersects based on l1.
    l1_l2 = ccw(l1_p1, l1_p2, l2_p1) * ccw(l1_p1, l1_p2, l2_p2)
    #// l2를 기준으로 l1이 교차하는 지 확인한다.
    #// Check if l1 intersects based on l2.
    l2_l1 = ccw(l2_p1, l2_p2, l1_p1) * ccw(l2_p1, l2_p2, l1_p2)

    ret = (l1_l2 < 0) and (l2_l1 < 0)

    return ret


def PolygonInOut(p, vertices):
    num_vertex = vertices.shape[0]

    #// 마지막 꼭지점과 첫번째 꼭지점이 연결되어 있지 않다면 오류를 반환한다.
    #// If the last vertex and the first vertex are not connected, an error is returned.
    if (vertices[0,0] != vertices[-1,0] or vertices[0,1] != vertices[-1,1]):
        print("Last vertex and first vertex are not connected.")
        return -1

    for i in range(num_vertex-1):
        #// 점 p가 i번째 꼭지점과 i+1번째 꼭지점을 이은 선분 위에 있는 경우
        #// If point p is on the line connecting the i and i + 1 vertices
        if( ccw(vertices[i,:], vertices[i+1,:], p) == 0 ):
            min_x = min(vertices[i, 0], vertices[i+1, 0])
            max_x = max(vertices[i, 0], vertices[i+1, 0])
            min_y = min(vertices[i, 1], vertices[i+1, 1])
            max_y = max(vertices[i, 1], vertices[i+1, 1])

            #// 점 p가 선분 내부의 범위에 있는 지 확인
            #// Determine if point p is in range within line segment
            if(min_x <= p[0] and p[0] <= max_x and min_y <= p[1] and p[1] <= max_y):
                return 1
    
    # // 다각형 외부에 임의의 점과 점 p를 연결한 선분을 만든다.
    # // Create a line segment connecting a random point outside the polygon and point p.
    outside_point = np.array([0.0,0.0])
    outside_point[0] = 1
    outside_point[1] = 1234567
    l1_p1 = outside_point
    l1_p2 = p

    ret = 0
    # // 앞에서 만든 선분과 다각형을 이루는 선분들이 교차되는 갯수가 센다.
    # // Count the number of intersections between the previously created line segments and the polygonal segments.
    for i in range(num_vertex-1):
        l2_p1 = vertices[i, :]
        l2_p2 = vertices[i+1, :]
        ret += LineIntersection(l1_p1, l1_p2, l2_p1, l2_p2)
    
    # // 교차한 갯수가 짝수인지 홀수인지 확인한다.
    # // Check if the number of crossings is even or odd.
    ret = ret % 2
    return ret


def IntersectionPoint(p1, p2, p3, p4):
    ret = np.array([0.0,0.0])
    ret[0] = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0]))/( (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]) )

    ret[1] = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / ( (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]) )

    return ret


#int interection_num;
#Point intersection_point[10];
def GetIntersection(points1, points2):
    num1 = points1.shape[0]
    num2 = points2.shape[0]

    # points1과 points2 각각을 반시계방향으로 정렬한다.
    # sort by counter clockwise point1 and points2.
    points1 = BubbleSort(points1)
    print(points1)

    #p = points2[0];
    points2 = BubbleSort(points2)

    #차례대로 점들을 이었을 때, 다각형이 만들어질 수 있도록 시작점을 마지막에 추가한다.
    #Add the starting point to the last in order to make polygon when connect points in order.
    points1 = np.append(points1, points1[:1,:], 0)
    points2 = np.append(points2, points2[:1,:], 0)

    #print(points1)
    #print(points2)

    # // points1의 다각형 선분들과 points2의 다각형 선분들의 교차점을 구한다.
    # // Find the intersection of the polygon segments of points1 and the polygon segments of points2.
    intersection_point = []
    for i in range(num1):
        l1_p1 = points1[i,:]
        l1_p2 = points1[i+1,:]
        for j in range(num2):
            l2_p1 = points2[j,:]
            l2_p2 = points2[j+1,:]

            # // 선분 l1과 l2가 교차한다면 교차점을 intersection_point에 저장한다.
            # // If line segments l1 and l2 intersect, store the intersection at intersection_point.
            if(LineIntersection(l1_p1, l1_p2, l2_p1, l2_p2)):
                intersection_point.append(IntersectionPoint(l1_p1, l1_p2, l2_p1, l2_p2))

#    print(intersection_point)

    for i in range(num1):
        if(PolygonInOut(points1[i,:], points2)):
            intersection_point.append(points1[i,:])

    for i in range(num2):
        if(PolygonInOut(points2[i,:], points1)):
            intersection_point.append(points2[i,:])

    # // restore
    # points1[num1].x = 0;    points1[num1].y = 0;
    # points2[num2].x = 0;    points2[num2].y = 0;

    #p = intersection_point[0];
    print(intersection_point)
    intersection_point = BubbleSort(np.array(intersection_point))

    ret = GetPolygonArea(intersection_point); 
    return ret


def GetIoU(points1, points2):
    interection_num = 0
    
    intersection_area = GetIntersection(points1, points2)
    A = GetPolygonArea(points1)
    B = GetPolygonArea(points2)
    union_area = A + B - intersection_area

    ret = intersection_area / union_area;       
    return ret

def main():

    points1 = np.array([[1.0, 2],
                        [3, 4],
                        [3, 1],
                        [1, 3],
                        [4, 2]])

    points2 = np.array([[2.0, 3],
                        [3, 2],
                        [5, 3],
                        [5, 4]])

    iou = GetIoU(points1, points2)
    print("IoU : ", iou);


if __name__=="__main__":
    main()