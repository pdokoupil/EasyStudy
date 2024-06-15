"""Add autoincrement

Revision ID: 88b1b92f50ef
Revises: 
Create Date: 2024-06-15 11:48:44.438813

"""
from alembic import op
import sqlalchemy as sa

from models import UserStudy

# revision identifiers, used by Alembic.
revision = '88b1b92f50ef'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Unfortunately it seems we cannot easily change column type to account for autoincrement in sqlite after the table has been created
    # We hack-around by creating temporary table with proper schema, paste all contents of userstudy table there
    # And replace userstudy with the newly created temporal
    op.execute("CREATE TABLE userstudy_tmp ( id INTEGER PRIMARY KEY AUTOINCREMENT, creator VARCHAR, guid VARCHAR, parent_plugin VARCHAR, settings VARCHAR, time_created DATETIME, active BOOLEAN, initialized BOOLEAN, initialization_error VARCHAR, FOREIGN KEY(creator) REFERENCES user (email) )")
    op.execute("""INSERT INTO userstudy_tmp(creator, guid, parent_plugin, settings, time_created, active, initialized, initialization_error)
                  SELECT creator, guid, parent_plugin, settings, time_created, active, initialized, initialization_error
                  FROM userstudy;""")
    op.execute("DROP TABLE userstudy;")
    op.execute("ALTER TABLE userstudy_tmp RENAME TO userstudy;")

    # Same hack is applied to all other tables that are currently using PRIMARY KEY relying on auto-increment feature
    # Participation
    op.execute("CREATE TABLE participation_tmp ( id INTEGER PRIMARY KEY AUTOINCREMENT, participant_email VARCHAR, age_group VARCHAR, gender VARCHAR, education VARCHAR, ml_familiar BOOLEAN, user_study_id INTEGER, time_joined DATETIME, time_finished DATETIME, uuid VARCHAR, language VARCHAR, extra_data VARCHAR, FOREIGN KEY(user_study_id) REFERENCES userstudy (id) )")
    op.execute("""INSERT INTO participation_tmp(participant_email, age_group, gender, education, ml_familiar, user_study_id, time_joined, time_finished, uuid, language, extra_data)
                  SELECT participant_email, age_group, gender, education, ml_familiar, user_study_id, time_joined, time_finished, uuid, language, extra_data
                  FROM participation;""")
    op.execute("DROP TABLE participation;")
    op.execute("ALTER TABLE participation_tmp RENAME TO participation;")

    # Message
    op.execute("CREATE TABLE message_tmp ( id INTEGER PRIMARY KEY AUTOINCREMENT, participation INTEGER, time DATETIME, data VARCHAR )")
    op.execute("""INSERT INTO message_tmp(participation, time, data)
                  SELECT participation, time, data
                  FROM message;""")
    op.execute("DROP TABLE message;")
    op.execute("ALTER TABLE message_tmp RENAME TO message;")

    # Interaction
    op.execute("CREATE TABLE interaction_tmp ( id INTEGER PRIMARY KEY AUTOINCREMENT, participation INTEGER, interaction_type VARCHAR, time DATETIME, data VARCHAR, FOREIGN KEY(participation) REFERENCES participation (id) )")
    op.execute("""INSERT INTO interaction_tmp(participation, interaction_type, time, data)
                  SELECT participation, interaction_type, time, data
                  FROM interaction;""")
    op.execute("DROP TABLE interaction;")
    op.execute("ALTER TABLE interaction_tmp RENAME TO interaction;")


def downgrade():
    op.execute("CREATE TABLE userstudy_tmp ( id INTEGER NOT NULL, creator VARCHAR, guid VARCHAR, parent_plugin VARCHAR, settings VARCHAR, time_created DATETIME, active BOOLEAN, initialized BOOLEAN, initialization_error VARCHAR, PRIMARY KEY (id), FOREIGN KEY(creator) REFERENCES user (email) )")
    op.execute("""INSERT INTO userstudy_tmp(creator, guid, parent_plugin, settings, time_created, active, initialized, initialization_error)
                  SELECT creator, guid, parent_plugin, settings, time_created, active, initialized, initialization_error
                  FROM userstudy;""")
    op.execute("DROP TABLE userstudy;")
    op.execute("ALTER TABLE userstudy_tmp RENAME TO userstudy;")

    op.execute("CREATE TABLE participation_tmp ( id INTEGER NOT NULL, participant_email VARCHAR, age_group VARCHAR, gender VARCHAR, education VARCHAR, ml_familiar BOOLEAN, user_study_id INTEGER, time_joined DATETIME, time_finished DATETIME, uuid VARCHAR, language VARCHAR, extra_data VARCHAR, PRIMARY KEY (id), FOREIGN KEY(user_study_id) REFERENCES userstudy (id) )")
    op.execute("""INSERT INTO participation_tmp(participant_email, age_group, gender, education, ml_familiar, user_study_id, time_joined, time_finished, uuid, language, extra_data)
                  SELECT participant_email, age_group, gender, education, ml_familiar, user_study_id, time_joined, time_finished, uuid, language, extra_data
                  FROM participation;""")
    op.execute("DROP TABLE participation;")
    op.execute("ALTER TABLE participation_tmp RENAME TO participation;")

    # Message
    op.execute("CREATE TABLE message_tmp ( id INTEGER NOT NULL, participation INTEGER, time DATETIME, data VARCHAR, PRIMARY KEY (id) )")
    op.execute("""INSERT INTO message_tmp(participation, time, data)
                  SELECT participation, time, data
                  FROM message;""")
    op.execute("DROP TABLE message;")
    op.execute("ALTER TABLE message_tmp RENAME TO message;")

    # Interaction
    op.execute("CREATE TABLE interaction_tmp ( id INTEGER NOT NULL, participation INTEGER, interaction_type VARCHAR, time DATETIME, data VARCHAR, PRIMARY KEY (id), FOREIGN KEY(participation) REFERENCES participation (id) )")
    op.execute("""INSERT INTO interaction_tmp(participation, interaction_type, time, data)
                  SELECT participation, interaction_type, time, data
                  FROM interaction;""")
    op.execute("DROP TABLE interaction;")
    op.execute("ALTER TABLE interaction_tmp RENAME TO interaction;")