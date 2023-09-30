#include "BTree.h"

int main()
{
  auto logger = Shtensor::Log::create("test");

  auto tree = Shtensor::BTree<int64_t>(4);

  tree.insert(0, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(1, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(2, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(3, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(4, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(5, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(6, 5);

  tree.insert(7, 5);

  tree.insert(8, 5);

  tree.insert(9, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(-1, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(-5, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(-8, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(-11, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.insert(-10, 5);

  fmt::print("{}\n", tree.make_pretty());

  tree.remove(-10);

  fmt::print("{}\n", tree.make_pretty());

  tree.remove(-11);
  tree.remove(-8); // right leaf rotation deletion

  tree.insert(-6,5);

  fmt::print("{}\n", tree.make_pretty());

  tree.remove(0); // left leaf rotation deletion

  fmt::print("{}\n", tree.make_pretty());

  tree.remove(-6); // leftmost leaf deletion with single index

  fmt::print("{}\n", tree.make_pretty());

  tree.remove(9);
  tree.remove(6); // right leaf deletion with single index

  fmt::print("{}\n", tree.make_pretty());

  // fill it up again for next tests
  tree.insert(9,0);
  tree.insert(12,0);
  tree.insert(13,0);
  tree.insert(10,0);
  tree.insert(11,0);

  fmt::print("{}\n", tree.make_pretty());

  tree.remove(9); // internal node with size > 1 and non-overflowing children after merge

  fmt::print("{}\n", tree.make_pretty());

  int result = 0;

  return result;
}