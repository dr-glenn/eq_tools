# filter quakeml input files and create output quakeml
import logging
import obspy
from obspy import UTCDateTime
from shapely.geometry import Point, Polygon     # for geographic selection
from shapely.ops import unary_union
import glob
from config import TEMPLATE_POLY,QUAKEML_DIR

BBOX = TEMPLATE_POLY
NZONE = 8   # equal sized zones within BBOX
    
def setup_logger(logname):
    import logging
    '''
    Messages will appear on console. Additional messages are written to a file for DEBUG usage.
    '''

    filehandler = logging.FileHandler(logname, mode='w')
    fileFormatter = logging.Formatter('%(asctime)s\t%(name)s - %(levelname)s: %(message)s')
    filehandler.setFormatter(fileFormatter)
    filehandler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    consoleFormatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
    console.setFormatter(consoleFormatter)
    console.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # required if ANY handler wants to display DEBUG msgs
    logger.addHandler(filehandler)
    logger.addHandler(console)
    
    return logger

logger = setup_logger('ev_select.log')

def create_zones(nz = NZONE):
    bound0 = (BBOX[0],BBOX[3])  # 0 is lower-left, 3 is lower-right
    bound2 = (BBOX[1],BBOX[2])  # 1 is upper-left, 2 is upper-right

    def lin_interp(start, end, n):
        lin = [start,]
        x0 = start[0]
        y0 = start[1]
        x1 = end[0]
        y1 = end[1]
        for i in range(1,n+1):
            x = x0 + i * (x1 - x0)/n
            y = y0 + i * (y1 - y0)/n
            lin.append((x,y))
        return lin
        
    lower = lin_interp(BBOX[0], BBOX[3], 8)
    upper = lin_interp(BBOX[1], BBOX[2], 8)
    logger.info('lower = {}'.format(lower))
    logger.info('upper = {}'.format(upper))

    # construct boxes clockwise
    def make_boxes(lower, upper):
        boxes = []
        for i in range(len(lower)-1):
            box = Polygon([lower[i], upper[i], upper[i+1], lower[i+1], lower[i]])
            boxes.append(box)
        return boxes

    boxes = make_boxes(lower,upper)  
    return boxes

boxes = create_zones()
logger.debug('boxes = {}'.format(boxes))
for idx,box in enumerate(boxes):
    logger.info('Region #{}: {}'.format(idx,box))


    
# Geographic filter based on arbitrary polygon (use to align with faults)
def polygonEvFilter(events, bbox, mag_lower=None):
    '''
    :param bbox: list of polygon vertices - each vertex is a tuple: (long,lat)
    :param events: Catalog of obspy events
    :param mag_lower: optionally remove events with smaller magnitude
    :return: Catalog of events insde polygon

    NOTE: coordinates are expected to be global (longitude,latitude)
    '''
    if not isinstance(bbox, Polygon):
        bbox = Polygon(bbox)    # better hope it's a list!
    evs = obspy.core.event.Catalog()
    for ev in events:
        # magnitude filter is complicated:
        #   some events have no magnitude (when found by EqTransformer)
        if ev.preferred_magnitude():
            ev_mag = ev.preferred_magnitude().mag
        else:
            ev_mag = None
        if mag_lower and ev_mag and (ev_mag < mag_lower):
            continue    # reject event
        else:
            pass        # allow event if ev_mag is None
        origin = ev.preferred_origin() or ev.origins[0]
        ll = Point(origin.longitude,origin.latitude)
        if ll.within(bbox):
            evs.append(ev)
    return evs

# I want to filter from QML file for events tha are found in zones 5,6,7 only.
box_select = [5,6,7]
zone = unary_union(boxes[5:8])
print('Selected Zone: {}'.format(zone))

qml_files = sorted(glob.glob(QUAKEML_DIR+'XO*.quakeml'))

cat_out = obspy.core.event.Catalog()
total_in = 0
total_out = 0

for qml in qml_files:
    logger.info('QuakeML file: {}'.format(qml))
    events = obspy.read_events(qml)
    evs = polygonEvFilter(events, zone)
    total_in += events.count()
    total_out += evs.count()
    cat_out.extend(evs)
    logger.info('{} events input, {} events after filter'.format(events.count(), evs.count()))
   
logger.info('total in = {}, total filtered = {}'.format(total_in, total_out))  
cat_out.write('ev_select.quakeml', format='QUAKEML') 

