import testing.service_helper_test as sht
import testing.service_logging_test as slt
import testing.service_orientations_test as sot


def test_services():
    sht.run_all()
    slt.run_all()
    sot.run_all()

test_services()
print("Everything passed.")