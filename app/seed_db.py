"""
seed_db.py - Populate the SQLite database with the initial university data.

Run once, after init_schema(). Idempotent: re-running clears the seed
tables first so the DB matches the current seed file. Feedback,
learned_patterns and chat_history are NOT cleared so user-taught
patterns survive re-seeding.
"""

import os
import sys

# Make 'app' importable when this is invoked directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_connection, init_schema  # noqa: E402


COURSES = [
    # (code, name, level, faculty, duration_years, fee_per_year, description)
    ('CS-BSC',  'BSc Computer Science',         'Undergraduate', 'Computing',   3.0, 3000,
     'Core CS programme covering algorithms, programming, networks, AI and software engineering.'),
    ('IT-BSC',  'BSc Information Technology',   'Undergraduate', 'Computing',   3.0, 3000,
     'Applied IT degree focusing on infrastructure, cloud, web platforms and IT operations.'),
    ('SE-BSC',  'BSc Software Engineering',     'Undergraduate', 'Computing',   3.0, 3100,
     'Engineering-led approach to large-scale software design, testing and project delivery.'),
    ('DS-BSC',  'BSc Data Science',             'Undergraduate', 'Computing',   3.0, 3200,
     'Statistics, machine learning and data engineering with industry case studies.'),
    ('CY-BSC',  'BSc Cyber Security',           'Undergraduate', 'Computing',   3.0, 3200,
     'Offensive and defensive security, cryptography, network defence and incident response.'),
    ('BBA',     'BBA (Business Administration)','Undergraduate', 'Business',    3.0, 3500,
     'General business degree spanning management, finance, marketing and entrepreneurship.'),
    ('MBA',     'MBA (Master of Business Administration)', 'Postgraduate', 'Business', 1.5, 5000,
     'Career-accelerating MBA with leadership, strategy and analytics modules.'),
    ('DS-MSC',  'MSc Data Science',             'Postgraduate',  'Computing',   1.0, 4500,
     'One-year intensive masters in data science with a 3-month industry project.'),
    ('CS-MSC',  'MSc Computer Science',         'Postgraduate',  'Computing',   1.0, 4500,
     'Advanced CS topics including ML systems, distributed computing and HCI.'),
]

FACULTY = [
    # (title, name, department, expertise, email, office_hours, is_dean)
    ('Prof.', 'David Reid',       'Administration',     'University leadership, strategy',
     'dean@university.edu', 'Tue & Thu 14:00-16:00', 1),
    ('Dr.',   'Sarah Johnson',    'Computer Science',   'Artificial Intelligence, Machine Learning',
     's.johnson@university.edu', 'Mon & Wed 10:00-12:00', 0),
    ('Prof.', 'Michael Chen',     'Computer Science',   'Software Engineering, Programming',
     'm.chen@university.edu', 'Tue 13:00-15:00', 0),
    ('Dr.',   'Emily Williams',   'Data Science',       'Database Systems, Big Data',
     'e.williams@university.edu', 'Mon 14:00-16:00', 0),
    ('Prof.', 'James Kumar',      'Cyber Security',     'Network Security, Cryptography',
     'j.kumar@university.edu', 'Wed 11:00-13:00', 0),
    ('Dr.',   'Lisa Anderson',    'Business',           'Marketing, Strategy',
     'l.anderson@university.edu', 'Thu 10:00-12:00', 0),
    ('Dr.',   'Ahmed Al-Hassan',  'Computer Science',   'Computer Networks, IoT',
     'a.alhassan@university.edu', 'Fri 09:00-11:00', 0),
    ('Dr.',   'Priya Sharma',     'Computer Science',   'Web Development, HCI',
     'p.sharma@university.edu', 'Mon 13:00-15:00', 0),
]

EVENTS = [
    # (name, start_date, end_date, location, category, description)
    ('Orientation Week',     '2026-01-12', '2026-01-16', 'Main Auditorium',
     'Academic',     'Welcome programme for all new students.'),
    ('Annual Hackathon',     '2026-02-21', '2026-02-22', 'Block C, Computing Lab',
     'Tech',         '24-hour coding competition open to all students.'),
    ('Career Fair 2026',     '2026-03-15', '2026-03-15', 'Main Hall',
     'Career',       'Top employers across IT, finance and consulting.'),
    ('Sports Day',           '2026-04-05', '2026-04-05', 'University Stadium',
     'Sports',       'Inter-faculty athletics and team sports.'),
    ('Cultural Festival',    '2026-05-10', '2026-05-12', 'Open-Air Theatre',
     'Cultural',     'Three-day festival with music, dance and food stalls.'),
    ('AI Research Symposium','2026-06-08', '2026-06-09', 'Block A, Lecture Hall 1',
     'Academic',     'Guest talks from leading AI researchers and student paper showcase.'),
    ('Open Day',             '2026-07-19', '2026-07-19', 'Whole Campus',
     'Outreach',     'Campus tour and Q&A for prospective students and parents.'),
]

EXAMS = [
    # (exam_type, start_date, end_date, format, notes)
    ('Midterm',         '2026-03-02', '2026-03-13', 'Mixed (written + online)',
     'Detailed timetable on student portal 2 weeks before start.'),
    ('Final',           '2026-06-15', '2026-06-30', 'Mostly written',
     'Results released within 3 weeks of completion.'),
    ('Supplementary',   '2026-08-03', '2026-08-10', 'Written',
     'For students who missed or failed an exam in the main sitting.'),
    ('Viva (PG only)',  '2026-07-06', '2026-07-10', 'Oral',
     'Postgraduate dissertation defence.'),
]

SCHOLARSHIPS = [
    # (name, max_percentage, eligibility, description)
    ('Merit Scholarship',     50, 'GPA 3.5+ at admission',
     'Awarded to high-achieving students each academic year.'),
    ('Need-Based Aid',        75, 'Documented financial need',
     'Means-tested support reviewed by the Financial Aid office.'),
    ('Sports Scholarship',    40, 'Represented region/country in a sport',
     'For students competing in intercollegiate or national sports.'),
    ('Early Bird Discount',   10, 'Applied before April deadline',
     'Flat 10% discount for students who finalise enrolment early.'),
    ('Research Excellence',   30, 'Published or strong research portfolio (PG only)',
     'Postgraduate scholarship recognising research achievement.'),
]

HOSTEL_ROOMS = [
    # (room_type, capacity, price_per_semester, amenities)
    ('Shared Twin Room',      2, 500,  'WiFi, study desk, shared bathroom, weekly cleaning'),
    ('Single Room',           1, 800,  'WiFi, en-suite bathroom, study desk, weekly cleaning'),
    ('Premium Single',        1, 1100, 'WiFi, en-suite, kitchenette, balcony, weekly cleaning'),
    ('Family Apartment',      4, 1500, 'Two bedrooms, full kitchen, living area - married/family students'),
]

KV_FACTS = [
    # (key, value, category)
    ('library_location',      'Block B, 2nd Floor',                                              'library'),
    ('library_hours_weekday', 'Monday-Friday 8:00 AM - 8:00 PM',                                 'library'),
    ('library_hours_weekend', 'Saturday 9:00 AM - 5:00 PM, Sunday closed',                       'library'),
    ('library_borrow_limit',  '5 books per student, 14-day loan',                                'library'),
    ('library_digital',       '24/7 e-book access via the student portal',                      'library'),

    ('contact_email',         'info@university.edu',                                             'contact'),
    ('contact_phone',         '+1-800-555-0199',                                                 'contact'),
    ('contact_address',       '123 University Avenue, Academic City',                            'contact'),
    ('contact_admin_office',  'Block A, Ground Floor (Mon-Fri 9:00 AM - 5:00 PM)',               'contact'),
    ('contact_support',       'support@university.edu',                                          'contact'),
    ('contact_emergency',     '+1-800-555-0911 (24/7 campus security)',                          'contact'),

    ('timetable_general',     'Classes run Monday-Friday, 8:00 AM to 4:00 PM',                   'timetable'),
    ('timetable_portal',      'Personal timetable available under "My Schedule" on the portal', 'timetable'),
    ('timetable_lecture_len', 'Lectures are typically 1-2 hours long',                           'timetable'),

    ('admission_portal',      'university.edu/apply',                                            'admission'),
    ('admission_fee',         '$25 application fee',                                             'admission'),
    ('admission_deadline',    'Sept intake closes Aug 1; Jan intake closes Dec 1',               'admission'),
    ('admission_documents',   'National ID, transcripts, personal statement, application form',  'admission'),
    ('admission_response',    'Decision within 2 weeks of complete submission',                  'admission'),

    ('grading_scale',         'A (90-100), B (80-89), C (70-79), D (60-69), F (<60)',            'exams'),
]


# Statements grouped by table so we can rebuild the seed without
# touching feedback / learned_patterns / chat_history.
SEED_TABLES = ('courses', 'faculty', 'events', 'exams',
               'scholarships', 'hostel_rooms', 'kv_facts')


def seed_all():
    init_schema()
    with get_connection() as conn:
        cur = conn.cursor()

        for table in SEED_TABLES:
            cur.execute(f"DELETE FROM {table}")

        cur.executemany(
            """INSERT INTO courses
               (code, name, level, faculty, duration_years, fee_per_year, description)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            COURSES
        )
        cur.executemany(
            """INSERT INTO faculty
               (title, name, department, expertise, email, office_hours, is_dean)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            FACULTY
        )
        cur.executemany(
            """INSERT INTO events
               (name, start_date, end_date, location, category, description)
               VALUES (?, ?, ?, ?, ?, ?)""",
            EVENTS
        )
        cur.executemany(
            """INSERT INTO exams
               (exam_type, start_date, end_date, format, notes)
               VALUES (?, ?, ?, ?, ?)""",
            EXAMS
        )
        cur.executemany(
            """INSERT INTO scholarships
               (name, max_percentage, eligibility, description)
               VALUES (?, ?, ?, ?)""",
            SCHOLARSHIPS
        )
        cur.executemany(
            """INSERT INTO hostel_rooms
               (room_type, capacity, price_per_semester, amenities)
               VALUES (?, ?, ?, ?)""",
            HOSTEL_ROOMS
        )
        cur.executemany(
            """INSERT INTO kv_facts (key, value, category)
               VALUES (?, ?, ?)""",
            KV_FACTS
        )

    print(f"Seeded {len(COURSES)} courses, {len(FACULTY)} faculty, "
          f"{len(EVENTS)} events, {len(EXAMS)} exams, "
          f"{len(SCHOLARSHIPS)} scholarships, {len(HOSTEL_ROOMS)} hostel rooms, "
          f"{len(KV_FACTS)} kv_facts.")


if __name__ == "__main__":
    seed_all()
