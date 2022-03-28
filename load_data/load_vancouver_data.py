import datetime
import json
import pandas as pd


def load_cultual_spaces():
    with open("vancouver-cultural-spaces.json") as data_file:    
        #print(data_file)
        data = json.load(data_file)
        #print(data)
        for line in data_file:
            data.append(json.loads(line))
        data_file.close()
        
        df = pd.DataFrame(data)
        
        drop_attributes = ['datasetid', 'recordid', 'record_timestamp']
        df.drop(drop_attributes, axis=1, inplace=True)
        #df = df[df['fields']] #change to area in yelp data set that we want to filter on.
        #print(df)
        
        # transform the list into a pandas DataFrame
        data_colnames = ['name', 'longitude', 'latitude'] 
        df_vancouver = pd.DataFrame(columns=data_colnames)
        
        for field in df['fields']:
            lon = field['geom']['coordinates'][0]
            lat = field['geom']['coordinates'][1]
            df_vancouver = df_vancouver.append({'name': field['cultural_space_name'], 'longitude': lon, 'latitude': lat}, ignore_index=True)

        #print(df_vancouver)
        return df_vancouver

def load_libraries():
    with open("vancouver-libraries.json") as data_file:    
        #print(data_file)
        data = json.load(data_file)
        #print(data)
        for line in data_file:
            data.append(json.loads(line))
        data_file.close()
        
        df = pd.DataFrame(data)
        
        drop_attributes = ['datasetid', 'recordid', 'record_timestamp']
        df.drop(drop_attributes, axis=1, inplace=True)
        #df = df[df['fields']] #change to area in yelp data set that we want to filter on.
        #print(df)
        
        # transform the list into a pandas DataFrame
        data_colnames = ['name', 'longitude', 'latitude'] 
        df_vancouver = pd.DataFrame(columns=data_colnames)
        
        for field in df['fields']:
             lon = field['geom']['coordinates'][0]
             lat = field['geom']['coordinates'][1]
             df_vancouver = df_vancouver.append({'name': field['name'], 'longitude': lon, 'latitude': lat}, ignore_index=True)

        #print(df_vancouver)
        return df_vancouver
        
def load_schools():
    with open("vancouver-schools.json") as data_file:    
        #print(data_file)
        data = json.load(data_file)
        #print(data)
        for line in data_file:
            data.append(json.loads(line))
        data_file.close()
        
        df = pd.DataFrame(data)
        
        drop_attributes = ['datasetid', 'recordid', 'record_timestamp']
        df.drop(drop_attributes, axis=1, inplace=True)
        #df = df[df['fields']] #change to area in yelp data set that we want to filter on.
        print(df)
        
        # transform the list into a pandas DataFrame
        data_colnames = ['name', 'longitude', 'latitude'] 
        df_vancouver = pd.DataFrame(columns=data_colnames)
        
        for field in df['fields']:
            lon = field['geom']['coordinates'][0]
            lat = field['geom']['coordinates'][1]
            df_vancouver = df_vancouver.append({'name': field['school_name'], 'longitude': lon, 'latitude': lat}, ignore_index=True)

#        print(df_vancouver)
        return df_vancouver
        
def main():
    df_vancouver = load_cultual_spaces()
    #print(df_vancouver)
    df_libraries = load_libraries()
    #print(df_libraries)
    df_schools = load_schools()
    #print(df_schools)

    for index, poi in df_libraries.iterrows():
        df_vancouver = df_vancouver.append({'name': poi['name'], 'longitude': poi['longitude'], 'latitude': poi['latitude']}, ignore_index=True)
        
    for index, poi in df_schools.iterrows():
        df_vancouver = df_vancouver.append({'name': poi['name'], 'longitude': poi['longitude'], 'latitude': poi['latitude']}, ignore_index=True)
    
    print(df_vancouver)
    df_vancouver.to_pickle('./df_vancouver_mixed' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl')
    
    
if __name__ == "__main__":
    main()
       