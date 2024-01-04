using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace GameLogic
{
    [CreateAssetMenu(fileName="Block", menuName="Gamelogic/Block", order=1)]
    public class BlockScriptableObject : ScriptableObject
    {
        // list of offsets from center of the block
        // (x,y) where origin in grid is lower left corner
        // positive x to the right, positive y up
        public List<Vector2Int> m_offsets;
        // this value is the coordinate tuple for the display of the block 
        // when it is in the list of placeable blocks
        // it will always be in a 5x5 grid!
        // it follows the coordinate system of the game, which is the mathematical one (origin down left, x to right, y up)
        public Vector2Int center;

        public List<Vector2Int> getBlockOffsets(){
            return m_offsets;
        }
        public Vector2Int getCenter(){
            return center;
        }
    }
}
