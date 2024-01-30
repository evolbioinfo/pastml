import unittest
from datetime import datetime

from pastml import numeric2datetime, datetime2numeric


class DateConversionTest(unittest.TestCase):

    def test_1_Jan_2020(self):
        d = datetime(2020, 1, 1)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 1 Jan 2020')

    def test_31_Dec_2020(self):
        d = datetime(2020, 12, 31)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 31 Dec 2020')

    def test_31_Dec_1994(self):
        d = datetime(1994, 12, 31)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 31 Dec 1994')

    def test_1_Jan_1995(self):
        d = datetime(1995, 1, 1)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 1 Jan 1995')

    def test_29_Feb_2020(self):
        d = datetime(2020, 2, 29)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 29 Feb 2020')

    def test_2_Feb_2020(self):
        d = datetime(2020, 2, 2)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 2 Feb 2020')

    def test_7_May_2020(self):
        d = datetime(2020, 5, 7)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 7 May 2020')

    def test_1_Jan_2021(self):
        d = datetime(2021, 1, 1)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 1 Jan 2021')

    def test_31_Dec_2021(self):
        d = datetime(2021, 12, 31)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 31 Dec 2021')

    def test_2_Feb_2021(self):
        d = datetime(2021, 2, 2)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 2 Feb 2021')

    def test_7_May_2021(self):
        d = datetime(2021, 5, 7)
        self.assertEqual(d, numeric2datetime(datetime2numeric(d)), msg='Was supposed to be 7 May 2021')