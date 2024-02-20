import os
import mysql.connector
port=3306
dbhost = 'btbrazwnstmtza82arcm-mysql.services.clever-cloud.com'
dbuser ='uu6tbaokps9llpam'
dbpass = '3bDQ3bD4upyn4t4o4iFi'
dbname = 'btbrazwnstmtza82arcm'
def get_database_connection():
    return mysql.connector.connect(
        host=dbhost,
        database=dbname,
        user=dbuser,
        password=dbpass,
        port=port
    )

def extract_and_append_to_database(image_folder_path, connection):
    try:
        cursor = connection.cursor()

        # List all folders in the given directory (image_folder_path)
        for folder_name in os.listdir(image_folder_path):
            folder_path = os.path.join(image_folder_path, folder_name)

            # Check if the item in the directory is a folder
            if os.path.isdir(folder_path):
                # Split folder name into student_name and student_id
                student_name, student_id = folder_name.split('_')

                # Append to the database
                insert_query = "INSERT INTO  vision_11_all_people (Name,person_id) VALUES (%s, %s)"
                values = (student_name, student_id)
                cursor.execute(insert_query, values)

        connection.commit()
        print("Data appended to the database successfully!")

    except Exception as e:
        print(f"Error in extract_and_append_to_database: {e}")

    finally:
        cursor.close()

# Example usage
image_folder_path = "Image_Attendance"
connection = get_database_connection()
extract_and_append_to_database(image_folder_path, connection)
connection.close()
