# Implement Haversine formulation for distance and bearing along great circle arcs.
# Also calculates for points not on the arc the distance perpendicular to the arc and nearest point on arc.
# See: https://www.movable-type.co.uk/scripts/latlong.html
# See: https://www.anycalculator.com/longitude.htm

import math
import numpy as np

def haversine(ll1, ll2):
    '''
    Haversine method for great circle distance between 2 points on earth.
    :param ll1: (longitude, latitude) numpy array in degrees
    :param ll2:  (longitude, latitude) numpy array in degrees
    :return: (a, c, d, bearing)
    a is the square of half the chord length
    c is the distance between points in radians
    d is the distance between points in km
    bearing is the initial bearing angle from ll1 to ll2
    '''
    # Using the haversine formula
    # phi is latitude in radians
    # theta is longitude in radians
    # convert to radians
    R = 6370.0
    ll1rad = math.pi/180.0 * ll1
    ll2rad = math.pi/180.0 * ll2
    ldelta = ll2rad - ll1rad
    phi_delta = ldelta[1]   # latitude in radians
    theta_delta = ldelta[0] # longitude in radians (refereneced paper uses lambda instead of theta)
    # c is the angular distance in radians; a is the square of half of the chord length
    a = math.sin(phi_delta/2)**2 + math.sin(theta_delta/2)**2 * math.cos(ll1rad[1]) * math.cos(ll2rad[1])
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c

    # initial bearing from ll1 to ll2
    y = math.sin(theta_delta) * math.cos(ll2rad[1])
    x = math.cos(ll1rad[1]) * math.sin(ll2rad[1]) - math.sin(ll1rad[1]) * math.cos(ll2rad[1]) * math.cos(theta_delta)
    theta = math.atan2(y, x)
    bearing = (theta * 180.0/math.pi +360) % 360

    return (a,c,d,theta)

def line_dist(ll1, ll2, ll3):
    '''
    Great circle defined by two long-lat points, ll1 and ll2.
    A point on the earth, ll3.
    Compute closest point on great circle from ll3.
    Compute distance along great circle and perpendicular distance from it.
    :param ll1: (longitude, latitude) one point on great circle
    :param ll2: (longitude, latitude) another point on great circle
    :param ll3: (longitude, latitude) point somewhere on earth
    :return: (dist along great circle, dist perpendicular) in km
    '''
    R = 6371E3     # radius of earth
    a12,c12,d12,bearing12 = haversine(ll1, ll2)
    a13,c13,d13,bearing13 = haversine(ll1, ll3)

    #dang_xt = math.asin(math.sin(c13) * math.sin(bearing13-bearing12))
    dang_xt = math.asin(math.sin(d13/R) * math.sin(bearing13-bearing12))
    d_xt = dang_xt * R  # cross-track dist (from LL3 to nearest point on LL1-LL2)
    dang_at = math.acos(math.cos(d13/R) / math.cos(dang_xt))
    d_at = dang_at * R  # along-track dist (to nearest point on LL1-LL2)
    return d_at,d_xt

if __name__ == '__main__':
    # run tests
    # test cases are confirmed using https://www.anycalculator.com/longitude.htm

    TEMPLATE_POLY = [ (-160.5,52.5), (-162.5,54.8), (-152.0,58.5), (-149.5,56.6), (-160.5,52.5) ]
    # southern box boundary
    endpt0 = np.asarray(TEMPLATE_POLY[0])
    endpt3 = np.asarray(TEMPLATE_POLY[3])
    # northern box boundary
    endpt1 = np.asarray(TEMPLATE_POLY[1])
    endpt2 = np.asarray(TEMPLATE_POLY[2])

    a12,c12,d12,bearing12 = haversine(endpt0, endpt3)
    print('South: {}, {}, {}, {}'.format(a12,c12,d12,180.0/math.pi*bearing12))

    a12,c12,d12,bearing12 = haversine(endpt1, endpt2)
    print('North: {}, {}, {}, {}'.format(a12,c12,d12,180.0/math.pi*bearing12))

    pt0 = endpt0
    dist,offset = line_dist(endpt0, endpt3, pt0)
    print('#0: dist={}, offset={}'.format(dist,offset))

    pt0 = endpt3
    dist,offset = line_dist(endpt0, endpt3, pt0)
    print('#3: dist={}, offset={}'.format(dist,offset))

    pt0 = endpt1
    dist,offset = line_dist(endpt0, endpt3, pt0)
    print('#1: dist={}, offset={}'.format(dist,offset))

    pt0 = endpt2
    dist,offset = line_dist(endpt0, endpt3, pt0)
    print('#2: dist={}, offset={}'.format(dist,offset))