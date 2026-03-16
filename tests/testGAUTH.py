from pipeline.notice_collection.collect import setupGoogleSheets


def testGAUTH():
    sheet = setupGoogleSheets()
    print("Sheet connected:", sheet)
    try:
        response = sheet.append_row(
            ["test", "test", "test", "test", "test", "test", "test", "test", "test"]
        )
        print("API Response:", response)
        print("Sheet row count after append:", sheet.row_count)
    except Exception as e:
        print(f"ERROR appending row: {e}")


if __name__ == "__main__":
    testGAUTH()
