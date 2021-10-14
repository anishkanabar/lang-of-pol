import pickle, os, sys #, pprint

### customization of tables ###
TABLE_NAME = "file_metadata"
TABLE_COLUMNS = "file_name varchar(200), file_length_seconds numeric, day int, month int, time int, year int, zone varchar(10)"
sql_table_creation = f"CREATE TABLE {TABLE_NAME} ({TABLE_COLUMNS});"

TABLE2_NAME = "vad_metadata_pydub"
TABLE2_COLUMNS = "file_name varchar(200), nonsilent_minutes numeric, nonsilent_slices int[]" # TODO add db, pydub to this statement
sql_table2_creation = f"CREATE TABLE {TABLE2_NAME} ({TABLE2_COLUMNS});"

TABLE3_NAME = "daily_metadata"
TABLE3_COLUMNS = "directory_name varchar(200), zone varchar(10), complete_data boolean, day_length_minutes numeric, files_total_silence int[], has_silent_files boolean"
sql_table3_creation = f"CREATE TABLE {TABLE3_NAME} ({TABLE3_COLUMNS});"
### filepaths/locations ###
zone = sys.argv[1]
sql_path = f"./{zone}.sql"
core_path = "/media/4tb/data/"
zone_path = os.path.join(core_path, zone)

### general functions ###
def sql_insertion_input(*args):
    """creates a line of the insert table command"""
    ENTRY = ', '.join([str(i) for i in args])
    return f"{ENTRY}"

def from_pickle_metafile(pickle_path, sql_path, metadata_endswith):
    """ converts a pickle file to an sql table """
    pkl_fil = open(pickle_path, 'rb')
    dictionary = pickle.load(pkl_fil)
    pkl_fil.close()
    for key in dictionary.keys():
        if key.endswith(".mp3") or key.endswith(".wav"):
            #file_location = os.path.join(os.path.split(pickle_path)[0], key)
            prop_dict = dictionary[key]
            #print(metadata_endswith)
            if metadata_endswith == "vad_dict.pkl":
                 generated_sql_statement = vad_insertion_from_properties(key, zone, prop_dict)
            elif metadata_endswith == "metadata_dict.pkl":
                generated_sql_statement = insertion_from_properties(key, zone, prop_dict)
            #print(metadata_endswith)
            with open(sql_path, "a") as fil:
                fil.write("".join(generated_sql_statement))
                fil.write("\n")

def date_folder_iter(base_path, zone, metadata_endswith, isdir=False):
    """ finds all the metadata pickle files in each date directory """
    available_folders = os.listdir(os.path.join(base_path, zone))
    for folder in available_folders:
        path_to_folder = os.path.join(base_path, zone, folder)
        for subdir,directory,files in os.walk(path_to_folder):
            for fil in files:
                if fil.endswith(metadata_endswith) and isdir==False:
                    pickle_path = os.path.join(base_path,zone,subdir,fil)
                    try:
                        #print(metadata_endswith)
                        from_pickle_metafile(pickle_path, sql_path, metadata_endswith)
                    except Exception as e:
                        pass
                elif fil.endswith(metadata_endswith) and isdir==True:
                    pickle_path = os.path.join(base_path,zone,subdir,fil)
                    daily_from_pickle_metafile(pickle_path, sql_path, metadata_endswith)
### individual_audiofile_metadata-specific functions ###
def insertion_from_properties(file_name, zone, prop_dict):
    """ uses mp3 properties to generate the insert line """
    recording_start_dict = prop_dict['recording_start']
    day = recording_start_dict['day']
    file_length_seconds = prop_dict.get('file_length_seconds', "error!")
    time = recording_start_dict['time']
    year = recording_start_dict['year']
    month = recording_start_dict['month']
    entries = sql_insertion_input(file_name, file_length_seconds, day, month, time, year, zone)
    return f"({entries}),"


### vad_metadata-specific functions ###
def vad_insertion_from_properties(file_name, zone, vad_prop_dict):
    """ uses mp3 properties to generate the insert line """
    vad_program = vad_prop_dict.get('pydub', "Error") # TODO figure out if we're using things other than pydub, then this will need to get modified
    db = vad_program.get(-24, "error") #TODO same as above this also probably needs to get modified
    nonsilent_minutes = db['nonsilent_minutes']
    nonsilent_slices = db["nonsilent_slices"]
    vad_entries = sql_insertion_input(file_name, nonsilent_minutes, nonsilent_slices) # TODO add db & pydub to this statement
    return f"({vad_entries}),"

### daily_audio_metadata-specific functions ###
def daily_insertion_from_properties(directory_location, zone, dictionary, metadata_endswith):
    try:
        day_length_minutes = dictionary["day_length_minutes"]
        files_total_silence = dictionary["files_total_silence"]
        complete_data = dictionary["complete_data"]
        has_silent_files = dictionary["has_silent_files"]
        directory_name = os.path.split(directory_location)[1]
        daily_entries = sql_insertion_input(directory_name, zone, day_length_minutes, files_total_silence, complete_data, has_silent_files)
        return f"({daily_entries}),"
    except:
        print(directory_location)

def daily_from_pickle_metafile(pickle_path, sql_path, metadata_endswith):
    pkl_fil = open(pickle_path, 'rb')
    dictionary = pickle.load(pkl_fil)
    pkl_fil.close()
    directory_location = os.path.split(pickle_path)[0]
    daily_generated_sql_statement = daily_insertion_from_properties(directory_location, zone, dictionary, metadata_endswith)
    try:
        with open(sql_path, "a") as fil:
            fil.write("".join(daily_generated_sql_statement))
            fil.write("\n")
    except Exception as e:
        pass


if __name__ == "__main__":
    with open(sql_path, "a") as fil:
        fil.write(sql_table_creation)
        fil.write("\n")
        fil.write(sql_table2_creation)
        fil.write("\n")
        fil.write(sql_table3_creation)
        fil.write("\n")
        fil.write(f"INSERT INTO {TABLE_NAME} VALUES")
        fil.write("\n")
    date_folder_iter(core_path, zone, "metadata_dict.pkl")
    with open(sql_path, "a") as fil:
        fil.write(";")
        fil.write("\n")
        fil.write(f"INSERT INTO {TABLE2_NAME} VALUES")
        fil.write("\n")
    date_folder_iter(core_path, zone, "vad_dict.pkl")
    with open(sql_path, "a") as fil:
        fil.write(";")
        fil.write("\n")
        fil.write(f"INSERT INTO {TABLE3_NAME} VALUES")
        fil.write("\n")
    date_folder_iter(core_path, zone, "metadata_dict.pkl", True)
    with open(sql_path, "a") as fil:
        fil.write(";")
