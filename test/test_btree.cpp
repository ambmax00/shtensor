#include "BTree.h"
#include "TestUtils.h"

#define CHECK_RETURN() \
  if (result) return result

int main()
{
  auto logger = Shtensor::Log::create("test");

  int result = 0;

  auto tree = Shtensor::BTree<int64_t>(4);

  tree.insert(0, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 1, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(1, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 2, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(2, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 3, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(3, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 4, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(4, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 5, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(5, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 6, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(6, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 7, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(7, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 8, result);

  CHECK_RETURN();

  tree.insert(8, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 9, result);

  CHECK_RETURN();

  tree.insert(9, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 10, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(-1, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 11, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(-5, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 12, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(-8, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 13, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(-11, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 14, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.insert(-10, 5);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 15, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(-10);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 14, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(-11);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 13, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(1); // right leaf rotation deletion
  
  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 12, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(9); // left leaf rotation

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 11, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  // fill it up well
  for (int i = 15; i < 50; ++i)
  {
    tree.insert(i,0);
  }

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 46, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(37); // delete internal node with one value

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 45, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(5); // delete internal node with two values

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 44, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(34);
  tree.erase(30);
  tree.erase(33);
  tree.erase(35);

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 40, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(36); // erase internal node with change of tree height

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 39, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(40); // erase internal node 

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 38, result);
  fmt::print("{}\n", tree.make_pretty());

  CHECK_RETURN();

  tree.erase(39); // erase internal node 

  SHTENSOR_TEST_TRUE(tree.is_valid(), result);
  SHTENSOR_TEST_EQUAL(tree.size(), 37, result);
  fmt::print("{}\n", tree.make_pretty());

  std::vector<int64_t> indices(tree.size());
  tree.fill_ordered_array(indices.data(), nullptr);

  fmt::print("Ordered array: ({})", fmt::join(indices.begin(), indices.end(), ","));

  for (auto iter = tree.begin(); iter != tree.end(); ++iter)
  {
    fmt::print("{}:{}\n", iter.key(), *iter);
  }
  

  return result;
}