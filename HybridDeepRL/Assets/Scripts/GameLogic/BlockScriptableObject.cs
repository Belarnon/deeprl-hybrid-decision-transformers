using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GameLogic
{
    [CreateAssetMenu(fileName="Block", menuName="Gamelogic/Block", order=1)]
    public class BlockScriptableObject : ScriptableObject
    {
        // list of offsets from center of the block
        // (x,y) where origin in grid is lower left corner
        // positive x to the right, positive y up
        public List<Vector2Int> offsets;
    }
}
