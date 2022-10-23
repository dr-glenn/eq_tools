# Manage families of earthquake matches.
import sys
import pandas as pd
import logging
from logging import StreamHandler,FileHandler
from logging.handlers import RotatingFileHandler
import templates
from my_util import dt_match

rlogger = logging.getLogger().setLevel(logging.NOTSET)      # root logger

#import my_logger
#logger = my_logger.setup_logger('families', 'families.log', level=logging.DEBUG)
defFormat = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
defFormatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
# must set root logger defaults before overriding with specific handlers
# This logger is used for running analysis programs, so precise datetime is not needed

log_file = 'families.log'
# console will show INFO messages, not DEBUG
consoleHandler = StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)
consoleFormatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
consoleHandler.setFormatter(consoleFormatter)
fileHandler = FileHandler(log_file, mode='w')
fileHandler.setFormatter(defFormatter)
fileHandler.setLevel(logging.DEBUG)

# File with info that does not have typical logging usage
printHandler = FileHandler('families.prt', mode='w')
printFormatter = logging.Formatter('%(message)s')
printHandler.setFormatter(printFormatter)
printHandler.setLevel(logging.INFO)
printLog = logging.getLogger('printLog')
printLog.addHandler(printHandler)

logger = logging.getLogger(__name__)
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

class Families:
    def __init__(self, match_df):
        '''

        :param match_df: DataFrame of all detection matches
        '''
        self.df = match_df
        self.families = None
        #self.family_df = None
        self.unmatched = 0  # count of templates that are detected by another template, but themselves have no detections
        #self.process()
        self.family()   # creates self.families
        logger.info('family_df with columns: {}'.format(self.families.columns.values))
        #logger.info(self.family_df)

    def export(self, filename, root_only=True):
        if not root_only:
            self.families.to_csv(filename)
        else:
            self.families[self.families['family_root']==-1].to_csv(filename)

    def _remove_dup_dt(self, dt_list, templ_dt):
        dt_ret = []
        if len(dt_list) > 1:
            dtl0 = sorted(list(set(dt_list)))
            dt_ret.append(dtl0[0])  # earliest time always included
            for i in range(len(dtl0)-1):
                if dt_match(dtl0[i], dtl0[i+1]) == False:
                    dt_ret.append(dtl0[i+1])    # times do not match, so keep it
        else:
            dt_ret = dt_list
        # Make sure that templ_dt is not in this list
        if templ_dt:
            dt_ret = [t for t in dt_ret if not dt_match(templ_dt,t)]
        return dt_ret

    def family(self):
        '''
        Create DataFrame with template time and detection times.
        Input is self.df: each row has templ_dt (template event time) and det_dt (new event detected time).
          In self.df, there may be multiple rows with same templ_dt value.
        Output: gather all detections for each template.
          Column 1 is templ_dt (template event time)
          Column 2 is old_dt, a list of all associated det_dt that match other templates
          Column 3 is new_dt, a list of all det_dt that are new events
        NOTE: there may be templates that detected other templates. These detections need to be
        consolidated into a single family - this is done later by method consolidate()
        :return: DataFrame of families
        '''
        # groupby will gather all rows that have same templ_dt value.
        # Each row apparently contains a DataFrame
        by_templ_dt = self.df.groupby(['templ_dt',])
        by_templ_list = []
        for tdt,frame in by_templ_dt:
            old_list = frame[frame['is_new']==0]['det_dt'].tolist()
            new_list = frame[frame['is_new']==1]['det_dt'].tolist()
            row = [tdt,old_list,new_list]   # each row in new DF
            by_templ_list.append(row)
        # creates DF from list. The index is integer 0 to N-1, first column is templ_dt, second is old_dt, third is new_dt
        fam_df = pd.DataFrame.from_records(by_templ_list, columns=['templ_dt','old_dt','new_dt'])
        # fam_df has one row for each template that detected any events, both old and new detections.
        # Each row represents a family, but the families may not be distinct.
        # Any templates that detected another template (old_dt) should be merged.
        self.families = fam_df
        before_csv = 'fam_selected.csv'
        logger.info('Before consolidate, write families to {}'.format(before_csv))
        self.families.to_csv(before_csv)
        n_templ = fam_df.shape[0]
        # templates that do not detect any other templates (old_dt events)
        only_new = fam_df[fam_df.apply(lambda x: len(x['old_dt'])==0, axis=1)]
        n_only_new = only_new.shape[0]
        logger.info('Before consolidate, {} families, {} match other templates, {} match only new'
                    .format(n_templ, n_templ-n_only_new,n_only_new))
        # Merge families whenever a template has detected another template
        self.consolidate()
        # Remove duplicates that result when families are merged and sort the events by time.
        self.post_fix()
        return self.families

    def getFamilies(self):
        return self.families[self.families['family_root'] == -1]

    def post_fix(self):
        '''
        After consolidate has run, events can have duplicate times. Also the family_root event should be the earliest.
        However, we don't have locations for new_dt events, so only use old_dt to find root.
        :return:
        '''
        for idx,row in self.families.iterrows():
            if row['family_root'] == -1:
                # templ_dt might not be earliest event!
                templ_dt = row['templ_dt']
                old_dt = row['old_dt']
                logger.debug('post_fix: templ_dt={}, old_dt={}'.format(templ_dt, old_dt))
                if len(old_dt) > 0:
                    old_dt.append(row['templ_dt'])
                    old_dt = self._remove_dup_dt(old_dt, None)
                    new_templ_dt = old_dt.pop(0)
                    if templ_dt != new_templ_dt:
                        logger.info('post_fix: event {} templ_dt={} updates to {}'.format(idx,templ_dt,new_templ_dt))
                    self.families.at[idx, 'templ_dt'] = new_templ_dt
                    # cannot use loc, it may throw "ValueError: Must have equal len keys and value when setting with an iterable"
                    # I did a lot of searching to understand this problem and finally discovered I must use 'at'.
                    # Still don't understand why.
                    self.families.at[idx, 'old_dt'] = old_dt
                if len(row['new_dt']) > 0:
                    self.families.at[idx, 'new_dt'] = sorted(list(set(row['new_dt'])))
            else:
                # leave the non-root members unchanged
                pass

    def consolidate(self):
        '''
        :param family_df: will be modified in place by this method
        :return:
        '''
        # iterate over the family_df
        # If template has old_dt values, then lookup each of those templates
        # If found, copy the old_dt template matches to both old_dt and new_dt of current template
        # If found, mark the found template so we know it has been copied to another family
        # When done iterating, select only templates that have not been marked and copy to new DF

        family_df = self.families
        # Whenever a template has been matched in old_dt, increment the count
        family_df['old_match_cnt'] = [0] * family_df.shape[0]
        family_df['family_root'] = [-1] * family_df.shape[0]     # each row might be a family root
        unmatched_dt = []   # any template that was found by another, but itself generated no matches
        fam_dict = dict()
        for idx,row in family_df.iterrows():
            # NOTE: scalar values are represented by value and are unchanged within iterrows.
            #  List values are obtained by reference and can be updated within iterrows.
            #  It is a very bad idea to update lists within the loop, because it can actually
            #  change the loop behavior!
            # So the rules are this:
            #  - if a scalar needs to be changed, use family_df.at for both reading and writing
            #  - if a scalar is not changed, it can be read from row value
            #  - if a list is changed, copy the list from row, update the copy, and at end of loop,
            #    update using family_df.at. This is necessary here because we have another loop inside
            #    the main loop; the list update must occur when all child loops are complete.
            iroot = family_df.at[idx,'family_root']
            #iroot    = row['family_root']
            templ_dt = row['templ_dt']
            old_dt   = row['old_dt'].copy()
            new_dt   = row['new_dt'].copy()
            n_old = len(old_dt)
            n_new = len(new_dt)
            logger.info('*** event {} ({}) with root={} has {} old matches and {} new matches'
                        .format(idx, templ_dt, iroot, n_old, n_new))
            if n_old == 0:
                # This template only has new detections, not other templates
                # If it is a root event, then we don't need to do anything
                # If it is not a root event, then it has already been copied to prior root
                logger.debug('event {} has 0 old detections and only {} new detections and is member of family {}'
                             .format(idx, n_new, iroot))
                if iroot == -1:
                    fam_dict[idx] = []
            else:
                # n_old > 0: This template has matches with other templates
                # Lookup each old_dt in family_df to find out if it is a template that itself generated matches
                is_root = False
                if iroot == -1:
                    is_root = True # this event qualifies as a family root, subject to change later
                    add_to = idx    # any child events will be copied to this event
                else:
                    # this event is already a member of another family, so it's old_dt will be copied
                    add_to = iroot # any child events will be copied to its family root
                    #logger.debug('  event {} is already member of {}'.format(idx,add_to))
                for match_dt in row['old_dt']:
                    # dt_match is True if match_dt within 30 seconds of some templ_dt
                    match_df = family_df[family_df.apply(lambda x: dt_match(x['templ_dt'], match_dt), axis=1)]
                    # TODO: Does it have at least one returned row?
                    if match_df.empty:
                        # TODO: Some templates do not generate any matches - they may not even match the event with templ_dt
                        #  that found it as a match. We find this strange, and would like an explanation!
                        logger.warning('no match for template {}, which was detected by {} at {}'.format(match_dt, idx, templ_dt))
                        self.unmatched += 1 # TODO: this is not correct way to tally unmatched
                        unmatched_dt.append(match_dt)   # this is a better way to find unmatched
                    else:
                        if match_df.shape[0] > 1:
                            logger.warning('{} has {} rows in match_df'.format(match_dt,match_df.shape[0]))
                        # don't expect more than one match, but could happen
                        for jdx,jrow in match_df.iterrows():
                            logger.debug('  match_dt: {} at index {:-3} is detected by {:-3}'.format(match_dt, jdx, idx))
                            # TODO: if not root, then skip, because this should already be copied?
                            # TODO: I think we have to ignore any matches where jdx < idx
                            if jdx < idx:   # we've already evaluated the jdx event
                                logger.debug('event {} detects previous event {}'.format(idx, jdx))
                                # TODO: However, if this event is root, we might add earlier event to it?
                                jroot = family_df.loc[jdx, 'family_root']
                                if is_root:     # idx is root
                                    if jroot == -1:
                                        # Both are root, but jdx event did not detect idx
                                        # Therefore jdx event should be child of idx
                                        logger.warning('{} is root, but did not detect {}'.format(jdx,idx))
                                    else:
                                        # TODO: this is a tricky case
                                        # jdx event is already child of another, but should also be child of idx
                                        logger.warning('{} is child of {}, but should be child of {}'.format(jdx,jroot,idx))
                                else:
                                    if jroot == -1:
                                        # idx is not root, but jdx is
                                        # Therefore jdx should be child of idx, is it?
                                        logger.info('SKIP: {} is root, but is marked as child of {}'.format(jdx,idx))
                                        continue
                                    else:
                                        # both are not root. Are they members of same family?
                                        logger.warning('{} and {} both not root, members of {} and {}'.format(jdx,idx,jroot,iroot))
                                #continue
                            # TODO: we are still considering cases where jdx < idx. Should we?
                            if family_df.loc[jdx, 'old_match_cnt'] == 0:
                                # copy match_dt matches to templ_dt row if this is the first time
                                # TODO: what should I do when copying old_dt? Copy those events' new_dt also?
                                old_dt.extend(jrow['old_dt'])
                                new_dt.extend(jrow['new_dt'])
                                logger.debug('  event {} has {} new, extend with {}'.format(add_to,n_new,len(jrow['new_dt'])))
                                n_new = len(new_dt)
                            else:
                                logger.debug('  {} ({}) already in a family, SKIP add to {}'.format(jdx,jrow['templ_dt'],templ_dt))
                            # mark the match to be included with another family
                            family_df.at[jdx, 'old_match_cnt'] += 1
                            if jdx == add_to:
                                logger.error('processing idx={}, add_to={}, jdx={}'.format(idx,add_to,jdx))
                            else:
                                logger.debug('  processing idx={}, add_to={}, jdx={}'.format(idx, add_to, jdx))
                            family_df.at[jdx, 'family_root'] = add_to
                        if is_root:
                            if add_to in fam_dict:
                                fam_dict[add_to].append(jdx)
                            else:
                                fam_dict[add_to] = [jdx]

            if idx != add_to:
                # Be sure to preserve old_dt and new_dt of the root
                old_dt.extend(family_df.at[add_to, 'old_dt'])
                new_dt.extend(family_df.at[add_to, 'new_dt'])
            logger.debug('  before remove_dup, old_dt={}'.format(old_dt))
            old_dt = self._remove_dup_dt(old_dt, None)
            logger.debug('   after remove_dup, old_dt={}'.format(old_dt))
            family_df.at[add_to, 'old_dt'] = old_dt

            logger.debug('  before remove_dup, new_dt={}'.format(new_dt))
            new_dt = self._remove_dup_dt(new_dt, None)
            logger.debug('   after remove_dup, new_dt={}'.format(new_dt))
            family_df.at[add_to, 'new_dt'] = new_dt

        logger.debug('consolidate: {} match times in unmatched_dt list'.format(len(unmatched_dt)))
        self.fam_dict = fam_dict

    def show(self):
        # TODO: still don't know if a template doesn't match its partner, but perhaps did match another,
        # because unmatched only counts templates that has zero matches.
        # Example: A detects both B and C. B detects only A. C detects only B. This is strange,
        # we expect that B should also detect C and C should also detect A.
        def _printit(idx, row):
            old_dt = [x.strftime('%Y-%m-%d %X') for x in row['old_dt']]
            new_dt = [x.strftime('%Y-%m-%d %X') for x in row['new_dt']]
            printLog.info('*{:-4}: root={:-3}, old_cnt={} {}\nold: {}\nnew: {}'.
                          format(idx, row['family_root'], row['old_match_cnt'], row['templ_dt'], old_dt, new_dt))
        def _print_fam_dict(fd):
            printLog.info('\nFamilies Dict: {} families'.format(len(fd)))
            n_old = 0   # families that contain multiple templates
            n_new = 0   # families that only detect new events
            for fam_dt in fd:
                if fd[fam_dt]:
                    n_old += 1
                else:
                    n_new += 1
                printLog.info('{} : {}'.format(fam_dt, fd[fam_dt]))
            printLog.info('families with multiple templates: {}, families with only new events: {}'.format(n_old, n_new))

        #df = self.families[self.families['family_root']==-1]
        df = self.families
        for idx,row in df.iterrows():
            _printit(idx,row)
        '''
        df = self.families[self.families['family_root']>=0]
        for idx,row in df.iterrows():
            _printit(idx,row)
        '''
        _print_fam_dict(self.fam_dict)
        #logger.info('{} templates were detected, but themselves generated no detections'.format(self.unmatched))
        logger.info('{} templates have matches'.format(self.families.shape[0]))

# run as main to test code
if __name__ == '__main__':
    import config
    import matches
    match_file = config.MATCH_RECORD_FILE
    match_obj = matches.MatchImages(None, None, match_file)
    match_obj.quality_stats()
    match_obj.filter(is_new=(0,1))  # defaults to quality=(3,4)
    my_df = match_obj.getSelectedDF()
    #my_df.to_csv('fam_selected.csv')
    logger.info('main: {} rows in filtered DF'.format(my_df.shape[0]))
    fams = Families(my_df)
    #fams.families.to_csv('fam_selected.csv')
    fams.show()
    fams.export('families.csv', root_only=False)
    #fam_df = fams.family()
    #print(fam_df)
