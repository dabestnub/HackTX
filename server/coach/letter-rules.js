const CONFUSABLE_LETTERS = {
  A: ["S", "T", "M"],
  B: ["D", "R", "U"],
  C: ["O", "G", "Q"],
  D: ["B", "F", "K"],
  E: ["S", "A", "O"],
  F: ["D", "O", "9"],
  G: ["Q", "C", "L"],
  H: ["U", "R", "V"],
  I: ["J", "Y", "L"],
  J: ["I", "Y", "L"],
  K: ["V", "P", "D"],
  L: ["G", "Y", "I"],
  M: ["N", "S", "A"],
  N: ["M", "T", "A"],
  O: ["C", "E", "F"],
  P: ["K", "Q", "G"],
  Q: ["G", "P", "C"],
  R: ["U", "V", "H"],
  S: ["A", "E", "M"],
  T: ["N", "A", "S"],
  U: ["V", "R", "H"],
  V: ["U", "K", "R"],
  W: ["V", "B", "R"],
  X: ["D", "K", "R"],
  Y: ["I", "J", "L"],
  Z: ["D", "X", "I"],
};

function getConfusableLetters(letter) {
  const target = String(letter || "").slice(0, 1).toUpperCase();
  const list = CONFUSABLE_LETTERS[target] || [];
  return list.filter((item) => /^[A-Z]$/.test(item));
}

module.exports = {
  getConfusableLetters,
};
