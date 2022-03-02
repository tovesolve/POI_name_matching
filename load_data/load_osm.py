import datetime
import osmium as osm
import pandas as pd
from pathlib import Path
import os.path

class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_data = []

    def tag_inventory(self, elem, elem_type):
        #print("elemtags: ", elem.tags)
        #print("elem:", elem)
        for tag in elem.tags:
            self.osm_data.append([elem_type, 
                                   elem.id, 
                                   elem.version,
                                   elem.visible,
                                   pd.Timestamp(elem.timestamp),
                                   elem.uid,
                                   elem.user,
                                   elem.location,
                                   len(elem.tags),
                                   tag.k, 
                                   tag.v])

    # Add only datapoint to osm_data which have a name-tag.
    def tag_inventory2(self, elem, elem_type):
        tag_list = []
        name = ''
        for tag in elem.tags:
            tag_list.append((tag.k, tag.v))
            if tag.k == 'name':
                name = str(tag.v)

        data = []
        if name != '':
            data.append(elem_type)
            #data.append(elem.id)
            #data.append(elem.version)
            #data.append(elem.visible)
            #data.append(pd.Timestamp(elem.timestamp))
            #data.append(elem.uid)
            #data.append(elem.user)
            data.append(name)
            data.append(elem.location)
            data.append(len(elem.tags))
            data.append(tag_list)
            self.osm_data.append(data)


    def node(self, n):
        self.tag_inventory2(n, "node")

    def way(self, w):
        #self.tag_inventory(w, "way")
        return

    def relatitudeion(self, r):
        #self.tag_inventory(r, "relatitudeion")
        return


osmhandler = OSMHandler()
# scan the input file and fills the handler list accordingly
osmhandler.apply_file('massachusetts-latest.osm')

# transform the list into a pandas DataFrame
data_colnames = ['type', 'name', 'location', 'len_tags', 'tags'] # 'id', 'version', 'visible', 'ts', 'uid', 'user',
df_osm = pd.DataFrame(osmhandler.osm_data, columns=data_colnames)

# filter by tag "name" (doesn't work with tag_inventory2)
#df_osm = df_osm[df_osm['tagkey'] == 'name']

# Split attribute location to two new attributes, longitude and latitude
splitter = lambda x: pd.Series([i for i in reversed(str(x).split('/'))])
coordinates = df_osm['location'].apply(splitter)
coordinates.rename(columns={0:'latitude',1:'longitude'},inplace=True)
coordinates = coordinates[['latitude', 'longitude']]
df_osm['latitude'] = coordinates['latitude']      # add column longitude to the df
df_osm['longitude'] = coordinates['longitude']      # add column latitude to the df
df_osm.drop(['location'], axis=1, inplace=True)

df_osm.to_pickle('./df_osm_ma' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl')


print(df_osm)
print('Number of rows in osm: ', df_osm.shape[0])
#print(coordinates)